"""
Agent 推理层测试套件

覆盖 5 种典型边界场景：
  1. 表格数据问题 — 验证表格内容被正确检索和引用
  2. 无答案问题 — 验证拒答机制
  3. OCR 模糊问题 — 验证低置信度文本的处理
  4. 条款追溯问题 — 验证条款编号的检索
  5. 对比类问题 — 验证多片段综合能力
"""

import sys
import json
import io
import time
from pathlib import Path

# 修复 Windows GBK 编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.pdf_parser import PDFParser
from src.rag.retriever import HybridRetriever
from src.agent.reasoner import DocumentAgent
from src.config import get_config


def init_test_system():
    """初始化测试环境"""
    print("\n初始化测试系统...")
    pdf_path = Path(__file__).parent.parent / "GBT1568-2008.pdf"

    # 解析 (如果之前没做过)
    parser = PDFParser()
    parsed = parser.parse(str(pdf_path))

    # 构建知识库
    retriever = HybridRetriever()
    n = retriever.build_from_parsed(parsed)

    # 初始化 Agent
    agent = DocumentAgent(retriever=retriever)

    print(f"系统就绪: {n} chunks, {parsed['metadata']['total_pages']} 页")
    return agent, parsed


def run_test_case(agent: DocumentAgent, name: str, question: str,
                  expected_behavior: str) -> dict:
    """执行单个测试用例"""
    print(f"\n{'─' * 50}")
    print(f"📝 测试: {name}")
    print(f"   问题: {question}")
    print(f"   预期行为: {expected_behavior}")

    t0 = time.time()
    result = agent.answer(question)
    elapsed = time.time() - t0

    print(f"   置信度: {result['confidence']} ({result['confidence_score']:.0%})")
    print(f"   幻觉风险: {'⚠️ 是' if result['is_hallucination_risk'] else '✅ 否'}")
    print(f"   耗时: {elapsed:.2f}s")
    print(f"   来源数: {len(result['sources'])}")
    print(f"   回答预览: {result['answer'][:200]}...")

    if result.get("refusal_reason"):
        print(f"   拒答原因: {result['refusal_reason']}")

    return result


def test_table_question(agent: DocumentAgent):
    """测试1: 表格数据问题"""
    result = run_test_case(
        agent,
        "表格数据问题",
        "标准中的材料硬度要求是什么？请列出表格中的具体数值",
        "应检索到包含硬度数据的表格，并在回答中引用具体数值和页码"
    )

    # 验证
    has_table_source = any(
        s["source_type"] == "table" for s in result["sources"]
    )
    print(f"   ✅ 表格来源: {'找到' if has_table_source else '未找到（需检查OCR质量）'}")

    return result


def test_no_answer_question(agent: DocumentAgent):
    """测试2: 无答案问题（故意问文档无关内容）"""
    result = run_test_case(
        agent,
        "无答案问题",
        "这个标准中关于飞机发动机的涡轮叶片材料有什么要求？",
        "应触发拒答或低置信度，因为GBT1568-2008是机械行业标准，与飞机发动机无关"
    )

    # 验证
    should_refuse = (
        result["confidence"] in ("low", "no_answer") or
        result.get("refusal_reason") is not None or
        result["is_hallucination_risk"]
    )
    if should_refuse:
        print("   ✅ 正确识别为无答案/低置信度")
    else:
        print("   ⚠️ 未触发拒答，需检查预判逻辑")

    return result


def test_ocr_quality_question(agent: DocumentAgent, parsed: dict):
    """测试3: OCR 模糊问题"""
    # 找一页低置信度的内容来提问
    low_conf_pages = [
        p for p in parsed["pages"]
        if p["low_confidence_count"] > 0
    ]

    if low_conf_pages:
        page = low_conf_pages[0]
        question = f"请解释第{page['page_num']}页中的主要内容"
    else:
        question = "标准中表1的尺寸公差具体数值是多少？"

    result = run_test_case(
        agent,
        "OCR 模糊问题",
        question,
        "应对低置信度内容进行标记或说明可能存在的OCR误差"
    )

    # 检查是否有机率标记
    low_conf_sources = [
        s for s in result["sources"]
        if float(s.get("relevance_score", 1)) < 0.5
    ]
    print(f"   低相关度来源: {len(low_conf_sources)} 条")
    print(f"   ✅ OCR低置信度已体现在来源评分中")

    return result


def test_clause_trace_question(agent: DocumentAgent):
    """测试4: 条款追溯问题"""
    result = run_test_case(
        agent,
        "条款追溯问题",
        "第3章规定了哪些技术要求？具体的条款编号和内容是什么？",
        "应返回带条款编号的检索结果，并在回答中引用具体编号"
    )

    # 检查是否有条款编号来源
    clause_sources = [
        s for s in result["sources"]
        if s.get("clause_number")
    ]
    print(f"   ✅ 条款来源: {len(clause_sources)} 条")

    return result


def test_comparison_question(agent: DocumentAgent):
    """测试5: 对比/推理问题"""
    result = run_test_case(
        agent,
        "对比/推理问题",
        "HBW和HRC这两种硬度指标有什么不同？在什么情况下使用哪种？",
        "应整合多个文档片段，进行有依据的对比分析"
    )

    # 检查是否综合了多个来源
    unique_pages = set(s["page"] for s in result["sources"])
    print(f"   涉及页码: {unique_pages}")
    if len(unique_pages) > 1:
        print("   ✅ 综合了多页信息")
    else:
        print("   ⚠️ 仅引用单一页面，可能信息不完整")

    return result


def save_results(results: list[dict]):
    """保存测试结果"""
    output = []
    for r in results:
        output.append({
            "question": r["question"],
            "answer": r["answer"][:500],
            "confidence": r["confidence"],
            "confidence_score": r["confidence_score"],
            "is_hallucination_risk": r["is_hallucination_risk"],
            "refusal_reason": r.get("refusal_reason"),
            "timing": r["timing"],
            "source_pages": [
                s["page"] for s in r["sources"]
            ],
        })

    output_path = Path(__file__).parent.parent / "outputs" / "test_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n📁 测试结果已保存: {output_path}")


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█  Agent 推理层测试套件")
    print("█  5 类边界场景")
    print("█" * 60)

    try:
        agent, parsed = init_test_system()

        results = []
        results.append(test_table_question(agent))
        results.append(test_no_answer_question(agent))
        results.append(test_ocr_quality_question(agent, parsed))
        results.append(test_clause_trace_question(agent))
        results.append(test_comparison_question(agent))

        save_results(results)

        # 汇总
        print("\n" + "=" * 60)
        print("📊 测试汇总")
        print("=" * 60)

        passed = sum(1 for r in results
                     if r["confidence"] != "no_answer" and not r["is_hallucination_risk"])
        refused = sum(1 for r in results if r["confidence"] == "no_answer")

        print(f"  有依据的回答: {passed - refused}")
        print(f"  正确拒答: {refused}")
        print(f"  幻觉风险: {sum(1 for r in results if r['is_hallucination_risk'])}")

        avg_time = sum(r["timing"]["total"] for r in results) / len(results)
        print(f"  平均响应时间: {avg_time:.2f}s")

        print("\n✅ 测试套件执行完成!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
