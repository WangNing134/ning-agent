"""
智能决策层 (Agent/Reasoner)：多阶段推理流水线

Pipeline:
  用户问题
    → [阶段1] 检索相关文档片段
    → [阶段2] 置信度预判 (是否有足够依据)
    → [阶段3] 答案生成 (DeepSeek)
    → [阶段4] 自校验 (幻觉检测 + 一致性验证)
    → 最终输出 (含置信度 + 来源引用)

拒答机制:
  - 阶段2 置信度预判发现无相关文档 → 直接拒答
  - 阶段4 自校验发现答案无依据 → 标记低置信度 / 拒答
"""

import json
import re
import time
from typing import Any

from openai import OpenAI
from loguru import logger

from src.config import get_config
from src.rag.retriever import HybridRetriever


class DocumentAgent:
    """智能文档问答 Agent"""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        config: dict | None = None,
    ):
        cfg = get_config()
        self.settings = cfg
        self.llm_cfg = cfg["llm"]
        self.agent_cfg = cfg["agent"]
        self.retriever = retriever

        # 初始化 DeepSeek 客户端 (OpenAI 兼容)
        api_key = self.llm_cfg.get("api_key", "")
        if not api_key:
            raise ValueError(
                "DeepSeek API Key 未设置! "
                "请在 .env 文件中设置 DEEPSEEK_API_KEY=sk-xxx"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.llm_cfg["api_base"],
            timeout=self.llm_cfg["timeout"],
        )

        logger.info(f"Agent 初始化完成 (LLM: {self.llm_cfg['model']})")

    # ─── 主入口 ────────────────────────────────────────────────

    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """
        处理用户问题，返回完整回答。

        Args:
            question: 用户问题
            top_k: 检索数量

        Returns:
            {
                "question": "...",
                "answer": "...",
                "confidence": "high|medium|low|no_answer",
                "confidence_score": 0.85,
                "sources": [{"page":1, "content":"...", "score":0.9}],
                "reasoning": "检索→预判→生成→自校验 过程",
                "is_hallucination_risk": false,
                "refusal_reason": null
            }
        """
        t0 = time.time()
        logger.info(f"收到问题: {question[:80]}...")

        # ===== 阶段1: 检索 =====
        t1 = time.time()
        retrieved = self.retrieve(question, top_k)
        retrieval_time = time.time() - t1
        logger.info(
            f"[阶段1] 检索: {len(retrieved)} 条结果 "
            f"(最高分: {retrieved[0]['combined_score']:.3f}) "
            f"耗时 {retrieval_time:.2f}s"
        )

        # ===== 阶段2: 置信度预判 =====
        t2 = time.time()
        pre_check = self._pre_check_confidence(question, retrieved)
        pre_check_time = time.time() - t2
        logger.info(f"[阶段2] 预判: {pre_check['verdict']} 耗时 {pre_check_time:.2f}s")

        if pre_check["verdict"] == "no_answer":
            return {
                "question": question,
                "answer": pre_check["refusal_message"],
                "confidence": "no_answer",
                "confidence_score": 0.0,
                "sources": retrieved[:3],
                "reasoning": self._build_reasoning(
                    "检索", retrieved, "预判-拒答", pre_check, None, None
                ),
                "is_hallucination_risk": False,
                "refusal_reason": pre_check.get("reason", "无相关文档"),
                "timing": {
                    "retrieval": retrieval_time,
                    "pre_check": pre_check_time,
                    "generation": 0,
                    "self_check": 0,
                    "total": time.time() - t0,
                },
            }

        # ===== 阶段3: 答案生成 =====
        t3 = time.time()
        generation = self._generate_answer(question, retrieved)
        gen_time = time.time() - t3
        logger.info(f"[阶段3] 生成: {len(generation['answer'])} 字符 耗时 {gen_time:.2f}s")

        # ===== 阶段4: 自校验 =====
        t4 = time.time()
        self_check = self._self_check(question, generation["answer"], retrieved)
        sc_time = time.time() - t4
        logger.info(f"[阶段4] 自校验: {self_check['verdict']} 耗时 {sc_time:.2f}s")

        # 最终置信度合并
        final_confidence = self._merge_confidence(pre_check, self_check)

        result = {
            "question": question,
            "answer": generation["answer"],
            "confidence": final_confidence["level"],
            "confidence_score": final_confidence["score"],
            "sources": self._format_sources(retrieved[:5]),
            "reasoning": self._build_reasoning(
                "检索", retrieved, "预判", pre_check,
                "生成", generation, "自校验", self_check,
            ),
            "is_hallucination_risk": self_check.get("is_hallucination", False),
            "refusal_reason": None,
            "timing": {
                "retrieval": retrieval_time,
                "pre_check": pre_check_time,
                "generation": gen_time,
                "self_check": sc_time,
                "total": time.time() - t0,
            },
        }

        logger.info(
            f"回答完成: 置信度={final_confidence['level']}, "
            f"总分={final_confidence['score']:.2f}, "
            f"总耗时={time.time()-t0:.2f}s"
        )

        return result

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """检索相关文档片段"""
        if self.retriever is None:
            raise RuntimeError("检索器未初始化! 请先调用 build_knowledge_base()")
        return self.retriever.retrieve(question, top_k)

    # ─── 阶段2: 置信度预判 ──────────────────────────────────────

    def _pre_check_confidence(
        self, question: str, retrieved: list[dict]
    ) -> dict:
        """
        预判检索结果是否足以回答问题。

        使用 keyword/rule-based 快速判断 + LLM 兜底。
        """
        # 快速规则判断
        if not retrieved:
            return {
                "verdict": "no_answer",
                "reason": "检索结果为空",
                "refusal_message": "根据现有文档，无法找到与该问题相关的任何依据。请确认问题是否与文档内容相关。",
                "confidence_score": 0.0,
            }

        top_score = retrieved[0].get("combined_score", 0)
        avg_score = sum(r.get("combined_score", 0) for r in retrieved[:3]) / min(len(retrieved), 3)

        # 快速过滤：最高相关性极低 → 直接拒答
        if top_score < 0.10 and avg_score < 0.05:
            return {
                "verdict": "no_answer",
                "reason": "检索相关性过低",
                "refusal_message": f"根据现有文档，关于「{question}」的内容，未找到任何相关依据。",
                "confidence_score": 0.0,
            }

        # 中低相关性 → LLM 二次预判
        if top_score < 0.45 or avg_score < 0.30:
            return self._llm_pre_check(question, retrieved)

        # 高相关性 → 确认通过
        return {
            "verdict": "proceed",
            "reason": f"检索相关度充足 (top={top_score:.3f}, avg={avg_score:.3f})",
            "confidence_score": min(top_score, 0.9),
        }

    def _llm_pre_check(self, question: str, retrieved: list[dict]) -> dict:
        """LLM 二次预判"""
        context_snippets = "\n---\n".join(
            f"[片段{i+1} 页码{r.get('page','?')}] {r['content'][:300]}"
            for i, r in enumerate(retrieved[:3])
        )

        prompt = f"""请判断以下文档片段是否能回答用户问题。

用户问题: {question}

文档片段:
{context_snippets}

请用JSON格式回答:
{{"can_answer": true/false, "reason": "简要说明", "relevant": true/false}}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content
            data = self._parse_json(text)
            if data and data.get("can_answer", True):
                return {"verdict": "proceed", "reason": data.get("reason", ""), "confidence_score": 0.6}
            else:
                return {
                    "verdict": "no_answer",
                    "reason": data.get("reason", "LLM判断无相关依据") if data else "无相关依据",
                    "refusal_message": f"根据现有文档，关于「{question}」的内容无法确认，建议查阅其他相关资料。",
                    "confidence_score": 0.1,
                }
        except Exception as e:
            logger.warning(f"LLM 预判失败: {e}，降级为 proceed")
            return {"verdict": "proceed", "reason": "LLM预判降级", "confidence_score": 0.5}

    # ─── 阶段3: 答案生成 ────────────────────────────────────────

    def _generate_answer(self, question: str, retrieved: list[dict]) -> dict:
        """基于检索结果生成答案"""

        # 构建上下文
        context_parts = []
        for i, r in enumerate(retrieved[:5]):
            src_label = (
                f"来源{i+1} [页码{r.get('page','?')}, "
                f"类型: {r.get('source_type', 'text')}"
            )
            if r.get("clause_number"):
                src_label += f", 条款: {r['clause_number']}"
            context_parts.append(f"{src_label}]\n{r['content']}")

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.agent_cfg["system_prompt"]},
            {"role": "user", "content": f"""请根据以下文档片段回答问题。

## 问题
{question}

## 文档片段
{context}

## 回答要求
1. 如果文档中有明确答案，直接引用并标注来源
2. 如果文档中只有部分信息，说明哪些是有依据的、哪些是推测的
3. 如果文档中没有答案，明确回复"根据现有文档，无法找到相关依据"
4. 对于表格数据，保持原始表格结构
5. 在回答末尾标注：置信度评估（高/中/低）和依据页码"""},
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.llm_cfg["model"],
                messages=messages,
                temperature=self.llm_cfg["temperature"],
                max_tokens=self.llm_cfg["max_tokens"],
            )
            answer = resp.choices[0].message.content.strip()
            return {"answer": answer, "model": self.llm_cfg["model"]}
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return {
                "answer": f"[生成失败: {e}] 请稍后重试。以下为检索到的原始片段供参考：\n\n{context[:500]}",
                "model": self.llm_cfg["model"],
                "error": str(e),
            }

    # ─── 阶段4: 自校验 ──────────────────────────────────────────

    def _self_check(
        self, question: str, answer: str, retrieved: list[dict]
    ) -> dict:
        """
        自校验：检查答案是否有依据、是否可能幻觉。

        校验维度：
          1. 答案的每个关键声明是否能在检索结果中找到支持
          2. 是否存在明显编造的数据（数字、日期、编号）
          3. 是否与检索到的原文存在矛盾
        """
        if not self.agent_cfg.get("enable_self_check", True):
            return {"verdict": "skipped", "is_hallucination": False, "score": 0.7}

        # 快速规则校验
        rules_check = self._rule_based_check(answer, retrieved)

        # LLM 深度校验
        llm_check = self._llm_self_check(question, answer, retrieved)

        # 合并
        is_hallucination = rules_check.get("hallucination", False) or llm_check.get("hallucination", False)
        score = (rules_check.get("score", 0.7) + llm_check.get("score", 0.7)) / 2

        if is_hallucination:
            score = min(score, 0.3)

        return {
            "verdict": "hallucination_risk" if is_hallucination else "verified",
            "is_hallucination": is_hallucination,
            "score": score,
            "rule_check": rules_check,
            "llm_check": llm_check,
        }

    def _rule_based_check(self, answer: str, retrieved: list[dict]) -> dict:
        """规则自校验：提取关键数字/编号，检查是否在检索结果中出现"""
        # 提取答案中的数字和编号
        numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
        clause_pattern = re.findall(r'(\d+(?:\.\d+)+)', answer)

        all_context = " ".join(r["content"] for r in retrieved)
        context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', all_context))

        # 检查哪些数字不在上下文中
        missing = numbers - context_numbers
        # 不检查简单的1位数（可能来自推理）
        suspicious = [n for n in missing if len(n) >= 3 or (len(n) >= 2 and "." in n)]

        score = max(0.0, 1.0 - len(suspicious) / max(len(numbers), 1))
        hallucination = len(suspicious) > len(numbers) * 0.3  # 超过30%的数字不在原文

        return {
            "total_numbers": len(numbers),
            "missing_numbers": len(suspicious),
            "suspicious_numbers": suspicious[:10],
            "hallucination": hallucination,
            "score": score,
        }

    def _llm_self_check(
        self, question: str, answer: str, retrieved: list[dict]
    ) -> dict:
        """LLM 自校验"""
        context = "\n---\n".join(r["content"][:500] for r in retrieved[:3])

        prompt = f"""你是一个严格的事实校验员。请判断以下回答是否完全基于提供的文档片段。

文档片段:
{context[:2000]}

用户问题: {question}

AI回答: {answer}

请检查:
1. 回答中的每个事实声明是否都能在文档片段中找到依据
2. 是否有编造的具体数字、日期或条款编号
3. 是否有超出文档范围的推测

用JSON格式回答:
{{"faithful": true/false, "has_fabrication": true/false, "fabricated_items": [], "score": 0.0-1.0}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.llm_cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            text = resp.choices[0].message.content
            data = self._parse_json(text)
            if data:
                return {
                    "hallucination": data.get("has_fabrication", False),
                    "faithful": data.get("faithful", True),
                    "score": data.get("score", 0.7),
                    "fabricated_items": data.get("fabricated_items", []),
                }
        except Exception as e:
            logger.warning(f"LLM 自校验失败: {e}")

        return {"hallucination": False, "faithful": True, "score": 0.5}

    # ─── 工具方法 ───────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """安全解析 JSON"""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 {...} 块
            m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    def _merge_confidence(self, pre_check: dict, self_check: dict) -> dict:
        """合并置信度"""
        pre_score = pre_check.get("confidence_score", 0.5)
        sc_score = self_check.get("score", 0.5)

        final_score = pre_score * 0.4 + sc_score * 0.6

        if final_score >= 0.7:
            level = "high"
        elif final_score >= 0.4:
            level = "medium"
        elif final_score > 0:
            level = "low"
        else:
            level = "no_answer"

        return {"score": round(final_score, 2), "level": level}

    def _format_sources(self, retrieved: list[dict]) -> list[dict]:
        """格式化来源引用"""
        sources = []
        for r in retrieved:
            sources.append({
                "page": r.get("page", "?"),
                "source_type": r.get("source_type", "text"),
                "clause_number": r.get("clause_number", ""),
                "content_preview": r["content"][:200],
                "relevance_score": r.get("combined_score", r.get("semantic_score", 0)),
            })
        return sources

    def _build_reasoning(self, *args) -> str:
        """构建推理过程描述"""
        parts = []
        for i in range(0, len(args), 2):
            label = args[i]
            data = args[i + 1] if i + 1 < len(args) else None
            if data is None:
                continue
            if isinstance(data, list):
                parts.append(
                    f"[{label}] 检索到 {len(data)} 条结果, "
                    f"最高相关度: {data[0].get('combined_score', 0):.3f}"
                )
            elif isinstance(data, dict):
                verdict = data.get("verdict", data.get("confidence_score", ""))
                parts.append(f"[{label}] {verdict}")
            else:
                parts.append(f"[{label}] {data}")
        return " → ".join(parts)
