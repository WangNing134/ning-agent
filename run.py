#!/usr/bin/env python
"""
保险文档智能问答 Agent — 主入口

用法:
  python run.py parse          # 仅解析 PDF
  python run.py ask "问题"     # 单次问答
  python run.py demo           # 交互式命令行演示
  python run.py api            # 启动 FastAPI 服务
  python run.py ui             # 启动 Streamlit 界面
  python run.py test           # 运行测试套件
"""

import sys
import io
import os
from pathlib import Path

# 修复 Windows GBK 编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保项目根目录在 Python path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from src.config import get_config
from src.parser.pdf_parser import PDFParser
from src.rag.retriever import HybridRetriever
from src.agent.reasoner import DocumentAgent
from loguru import logger


def cmd_parse():
    """解析 PDF"""
    pdf_path = "GBT1568-2008.pdf"
    parser = PDFParser()
    result = parser.parse(pdf_path)

    meta = result["metadata"]
    print(f"\n✅ 解析完成!")
    print(f"   总页数: {meta['total_pages']}")
    print(f"   结构化条目: {len(result['structured_entries'])}")
    print(f"   表格: {meta['pages_with_tables']} 页含表格")
    print(f"   Markdown: outputs/{Path(pdf_path).stem}_parsed.md")


def cmd_ask(question: str):
    """单次问答"""
    pdf_path = "GBT1568-2008.pdf"

    # 初始化
    print("正在初始化系统...")
    parser = PDFParser()
    parsed = parser.parse(pdf_path)

    retriever = HybridRetriever()
    retriever.build_from_parsed(parsed)

    agent = DocumentAgent(retriever=retriever)

    # 问答
    print(f"\n❓ 问题: {question}")
    print("━" * 60)
    result = agent.answer(question)
    print(result["answer"])
    print("━" * 60)
    print(f"置信度: {result['confidence']} ({result['confidence_score']:.0%})")
    print(f"幻觉风险: {'⚠️ 是' if result['is_hallucination_risk'] else '✅ 否'}")
    print(f"耗时: {result['timing']['total']:.2f}s")


def cmd_demo():
    """交互式命令行演示"""
    print("\n" + "🏦" * 30)
    print("   保险文档智能问答 Agent — 交互演示")
    print("🏦" * 30 + "\n")

    pdf_path = "GBT1568-2008.pdf"

    # 初始化
    print("⏳ 初始化系统 (解析 + 索引)...")
    parser = PDFParser()
    parsed = parser.parse(pdf_path)
    print(f"   ✅ PDF 解析: {parsed['metadata']['total_pages']} 页")

    retriever = HybridRetriever()
    n = retriever.build_from_parsed(parsed)
    print(f"   ✅ 知识库: {n} chunks")

    agent = DocumentAgent(retriever=retriever)
    print("   ✅ Agent 就绪\n")

    # 预设问题
    presets = [
        "GB/T 1568-2008 标准的全称是什么？",
        "标准中材料硬度有哪些技术要求？",
        "标准中有哪些表格？请列举",
        "这个标准规定了哪些检验方法？",
        "[测试-无答案] 这个标准中关于航空发动机的要求是什么？",
    ]

    print("📋 输入问题（或输入 1-5 选择预设问题，q 退出）:\n")
    for i, q in enumerate(presets, 1):
        print(f"  [{i}] {q}")
    print()

    while True:
        try:
            user_input = input("🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("👋 再见!")
            break
        if user_input.isdigit() and 1 <= int(user_input) <= len(presets):
            question = presets[int(user_input) - 1]
        else:
            question = user_input

        print(f"\n⏳ 思考中...")
        result = agent.answer(question)

        print(f"\n{'━' * 60}")
        print(result["answer"])
        print(f"{'━' * 60}")
        print(f"📊 置信度: {result['confidence']} ({result['confidence_score']:.0%}) | "
              f"幻觉风险: {'⚠️' if result['is_hallucination_risk'] else '✅'} | "
              f"耗时: {result['timing']['total']:.2f}s")
        print(f"📚 来源: {len(result['sources'])} 条")
        print()


def cmd_api():
    """启动 FastAPI 服务"""
    import uvicorn
    cfg = get_config()
    print(f"启动 API 服务: http://{cfg['api']['host']}:{cfg['api']['port']}")
    print(f"API 文档: http://{cfg['api']['host']}:{cfg['api']['port']}/docs")
    uvicorn.run(
        "src.api.server:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=False,
        log_level="info",
    )


def cmd_ui():
    """启动 Streamlit 界面"""
    import subprocess
    cfg = get_config()
    print(f"启动 Streamlit 界面: http://{cfg['ui']['host']}:{cfg['ui']['port']}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "src" / "api" / "ui.py"),
        "--server.port", str(cfg["ui"]["port"]),
        "--server.address", cfg["ui"]["host"],
    ])


def cmd_test():
    """运行所有测试"""
    print("\n🧪 运行测试套件...\n")

    # 先测试解析层
    print("═" * 60)
    print("  1/2 解析层测试")
    print("═" * 60)
    import subprocess
    subprocess.run([
        sys.executable,
        str(Path(__file__).parent / "tests" / "test_parser.py"),
    ])

    print("\n═" * 60)
    print("  2/2 Agent 推理层测试")
    print("═" * 60)
    subprocess.run([
        sys.executable,
        str(Path(__file__).parent / "tests" / "test_agent.py"),
    ])

    print("\n✅ 全部测试完成!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "parse":
        cmd_parse()
    elif cmd == "ask":
        question = sys.argv[2] if len(sys.argv) > 2 else input("请输入问题: ")
        cmd_ask(question)
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "api":
        cmd_api()
    elif cmd == "ui":
        cmd_ui()
    elif cmd == "test":
        cmd_test()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
        sys.exit(1)
