"""
知识检索层：混合检索 (Hybrid Search)

检索策略（无需本地 Embedding 模型）：
  1. 关键词检索 (BM25): 精确关键词匹配，中文 2-gram 分词
  2. DeepSeek 重排序: 对 Top-K 候选进行语义相关性打分
  3. 混合融合: 加权合并两种结果

每个 Chunk 携带 Metadata:
  - page: 页码
  - source_type: text / clause / table
  - confidence: OCR 置信度
  - clause_number: 条款编号 (如适用)

设计取舍:
  - 不用本地 Embedding 模型，避免下载大文件
  - 用 DeepSeek API 做语义理解（已用于生成，复用成本低）
  - BM25 保证召回率，DeepSeek 保证精确率
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from openai import OpenAI

from src.config import get_config


class HybridRetriever:
    """混合检索器：BM25 关键词 + DeepSeek 语义重排序"""

    def __init__(self, config: dict | None = None):
        cfg = get_config()
        self.settings = cfg
        self.db_cfg = cfg["vectordb"]
        self.chunk_cfg = cfg["chunking"]
        self.llm_cfg = cfg["llm"]
        self.hybrid_alpha = cfg["vectordb"]["hybrid_alpha"]

        # DeepSeek 客户端 (用于重排序)
        api_key = self.llm_cfg.get("api_key", "")
        if not api_key:
            api_key = cfg.get("llm", {}).get("api_key", "")
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=self.llm_cfg["api_base"],
            timeout=30,
        )

        # 文档存储（内存）
        self._docs: dict[str, dict] = {}  # doc_id -> chunk info
        self._all_chunks: list[dict] = []

        # BM25 索引
        self._bm25_index: dict[str, dict[str, int]] = {}
        self._bm25_avg_dl: float = 0
        self._bm25_N: int = 0

        logger.info("检索器初始化完成 (BM25 + DeepSeek 重排序)")

    # ─── 构建知识库 ───────────────────────────────────────────

    def build_from_parsed(self, parsed_result: dict) -> int:
        """
        从解析结果构建知识库。

        Args:
            parsed_result: PDFParser.parse() 的输出

        Returns:
            入库的 chunk 数量
        """
        entries = parsed_result["structured_entries"]
        logger.info(f"开始构建知识库: {len(entries)} 个条目")

        chunks = self._chunk_entries(entries)
        logger.info(f"切片完成: {len(chunks)} 个 chunk")

        if not chunks:
            logger.warning("没有有效的数据块，跳过知识库构建")
            return 0

        # 存储文档
        self._docs.clear()
        for c in chunks:
            self._docs[c["chunk_id"]] = c
        self._all_chunks = chunks

        # 构建 BM25 索引
        self._build_bm25(chunks)

        logger.info(f"知识库构建完成: {len(chunks)} chunks 已索引")
        return len(chunks)

    def _chunk_entries(self, entries: list[dict]) -> list[dict]:
        """将结构化条目切片"""
        chunks = []
        chunk_size = self.chunk_cfg["chunk_size"]
        chunk_overlap = self.chunk_cfg["chunk_overlap"]

        for entry in entries:
            content = entry["content"]
            if len(content) <= chunk_size:
                chunks.append({
                    **entry,
                    "chunk_id": f"p{entry['page']}_{len(chunks)}",
                })
            else:
                start = 0
                while start < len(content):
                    end = min(start + chunk_size, len(content))
                    chunk_text = content[start:end]
                    chunks.append({
                        **{k: v for k, v in entry.items() if k != "content"},
                        "content": chunk_text,
                        "chunk_id": f"p{entry['page']}_s{start}_{len(chunks)}",
                        "chunk_start": start,
                        "chunk_end": end,
                    })
                    start += chunk_size - chunk_overlap

        return chunks

    def _build_bm25(self, chunks: list[dict]) -> None:
        """构建简易 BM25 索引"""
        self._bm25_index.clear()
        total_len = 0

        for chunk in chunks:
            doc_id = chunk["chunk_id"]
            words = self._tokenize(chunk["content"])
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            self._bm25_index[doc_id] = word_freq
            total_len += len(words)

        self._bm25_N = len(chunks)
        self._bm25_avg_dl = total_len / max(self._bm25_N, 1)
        logger.debug(f"BM25 索引: {self._bm25_N} 文档, 平均长度 {self._bm25_avg_dl:.0f}")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单混合分词（中文2-gram + 英文单词 + 数字 + 标点关键符号）"""
        # 英文单词
        eng_words = re.findall(r'[a-zA-Z]+', text)
        # 中文 2-gram
        chinese = re.sub(r'[^一-鿿]', '', text)
        cn_grams = [chinese[i:i+2] for i in range(len(chinese)-1)] if len(chinese) > 2 else list(chinese)
        # 数字/编号
        numbers = re.findall(r'\d+(?:\.\d+)*', text)
        # 标准号 (如 GB/T 1568)
        std_ids = re.findall(r'[A-Z]+/\w+\s+\d+', text)
        return eng_words + cn_grams + numbers + std_ids

    # ─── 检索 ──────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int | None = None, use_rerank: bool = True
    ) -> list[dict[str, Any]]:
        """
        混合检索。

        Args:
            query: 用户问题
            top_k: 返回数量
            use_rerank: 是否使用 DeepSeek 重排序

        Returns:
            [{"content": "...", "page": 1, "source_type": "text",
              "keyword_score": 0.8, "semantic_score": 0.9,
              "combined_score": 0.85, ...}]
        """
        top_k = top_k or self.db_cfg["retrieval_k"]
        alpha = self.hybrid_alpha

        # 1. BM25 关键词检索（召回 top_k * 3）
        keyword_results = self._bm25_search(query, top_k * 3)
        logger.debug(f"BM25 召回: {len(keyword_results)} 条")

        if not keyword_results:
            logger.warning(f"BM25 无结果: '{query[:60]}...'")
            return []

        # 2. DeepSeek 重排序 (语义打分)
        semantic_scores: dict[str, float] = {}
        if use_rerank and len(keyword_results) > 1:
            semantic_scores = self._deepseek_rerank(query, keyword_results[:top_k * 2])
            logger.debug(f"DeepSeek 重排序: {len(semantic_scores)} 条打分")

        # 3. 融合
        combined = self._merge_results(keyword_results, semantic_scores, alpha, top_k)

        logger.debug(
            f"检索 '{query[:50]}...' → {len(combined)} 结果 "
            f"(BM25: {len(keyword_results)}, 重排序: {len(semantic_scores)})"
        )

        return combined

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 关键词检索"""
        k1, b = 1.5, 0.75
        query_terms = self._tokenize(query)
        scores = {}

        for doc_id, word_freq in self._bm25_index.items():
            score = 0
            doc_len = sum(word_freq.values())
            for term in query_terms:
                if term in word_freq:
                    tf = word_freq[term]
                    n_t = sum(1 for wf in self._bm25_index.values() if term in wf)
                    idf = max(0, np.log((self._bm25_N - n_t + 0.5) / (n_t + 0.5) + 1))
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_len / max(self._bm25_avg_dl, 1))
                    score += idf * numerator / max(denominator, 0.01)
            if score > 0:
                scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in sorted_docs:
            doc_info = self._docs.get(doc_id, {})
            results.append({
                "doc_id": doc_id,
                "content": doc_info.get("content", ""),
                "keyword_score": round(score, 4),
                "semantic_score": 0,
                "combined_score": round(score, 4),
                "page": doc_info.get("page", "?"),
                "source_type": doc_info.get("type", "text"),
                "clause_number": doc_info.get("clause_number", ""),
                "confidence": doc_info.get("confidence", 0),
            })
        return results

    def _deepseek_rerank(self, query: str, candidates: list[dict]) -> dict[str, float]:
        """
        调用 DeepSeek 对候选文档进行语义相关性打分。

        返回: {doc_id: relevance_score}
        """
        # 构造 reranking prompt
        items_text = []
        for i, c in enumerate(candidates):
            snippet = c["content"][:300].replace("\n", " ")
            items_text.append(f"[{i}] 页码{c.get('page','?')}: {snippet}")

        prompt = f"""请对以下文档片段与用户问题的相关性打分。

用户问题: {query}

文档片段:
{chr(10).join(items_text)}

请用 JSON 格式返回每个片段的相关性分数(0-1):
{{"scores": [{{"index": 0, "score": 0.85, "reason": "简要理由"}}, ...]}}

只返回 JSON，不要其他内容。"""

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_cfg["model"],
                messages=[
                    {"role": "system", "content": "你是一个文档相关性评估器。只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=500,
            )
            text = resp.choices[0].message.content

            # 解析 JSON
            data = self._parse_json(text)
            if data and "scores" in data:
                scores = {}
                for item in data["scores"]:
                    idx = item.get("index", 0)
                    if 0 <= idx < len(candidates):
                        doc_id = candidates[idx]["doc_id"]
                        scores[doc_id] = float(item.get("score", 0.5))
                return scores
        except Exception as e:
            logger.warning(f"DeepSeek 重排序失败: {e}")

        return {}

    def _merge_results(
        self,
        keyword_results: list[dict],
        semantic_scores: dict[str, float],
        alpha: float,
        top_k: int,
    ) -> list[dict]:
        """融合 BM25 和语义分数"""
        if not semantic_scores:
            # 无重排序结果，直接用 BM25
            for r in keyword_results:
                r["combined_score"] = r["keyword_score"]
            keyword_results.sort(key=lambda x: x["combined_score"], reverse=True)
            return keyword_results[:top_k]

        # 归一化 BM25 分数
        max_bm25 = max(r["keyword_score"] for r in keyword_results) if keyword_results else 1
        for r in keyword_results:
            r["keyword_score"] = r["keyword_score"] / max(max_bm25, 0.01)

        # 归一化语义分数
        max_sem = max(semantic_scores.values()) if semantic_scores else 1
        for doc_id in semantic_scores:
            semantic_scores[doc_id] = semantic_scores[doc_id] / max(max_sem, 0.01)

        # 合并
        for r in keyword_results:
            sem_score = semantic_scores.get(r["doc_id"], 0)
            r["semantic_score"] = round(sem_score, 4)
            r["combined_score"] = round(
                alpha * sem_score + (1 - alpha) * r["keyword_score"], 4
            )

        keyword_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return keyword_results[:top_k]

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """安全解析 JSON（处理 markdown 代码块包裹）"""
        if not text:
            return None
        # 去除 markdown 代码块
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"scores"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    def collection(self):
        """兼容旧 API：返回文档数量"""
        class CollectionProxy:
            def __init__(self, parent):
                self._parent = parent
            def count(self):
                return len(self._parent._docs)
        return CollectionProxy(self)
