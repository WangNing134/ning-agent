"""
Streamlit 前端界面

提供交互式文档问答体验。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from loguru import logger

from src.config import get_config
from src.parser.pdf_parser import PDFParser
from src.rag.retriever import HybridRetriever
from src.agent.reasoner import DocumentAgent


# ─── 页面配置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="保险文档智能问答 Agent",
    page_icon="📄",
    layout="wide",
)

# ─── Session State 初始化 ──────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.question_history = []
    st.session_state.parsed_result = None
    st.session_state.retriever = None
    st.session_state.agent = None


def init_system(pdf_path: str):
    """初始化系统"""
    with st.spinner("正在解析 PDF..."):
        parser = PDFParser()
        st.session_state.parsed_result = parser.parse(pdf_path)

    with st.spinner("正在构建知识库..."):
        st.session_state.retriever = HybridRetriever()
        st.session_state.retriever.build_from_parsed(
            st.session_state.parsed_result
        )

    with st.spinner("正在初始化 Agent..."):
        st.session_state.agent = DocumentAgent(
            retriever=st.session_state.retriever
        )

    st.session_state.initialized = True


# ─── 侧边栏 ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 保险文档问答")
    st.markdown("---")

    pdf_path = st.text_input(
        "PDF 路径",
        value="GBT1568-2008.pdf",
    )

    if st.button("🚀 初始化系统", type="primary"):
        if Path(pdf_path).exists():
            init_system(pdf_path)
        else:
            st.error(f"文件不存在: {pdf_path}")

    if st.session_state.initialized:
        meta = st.session_state.parsed_result["metadata"]
        st.success(f"✅ 系统就绪")
        st.markdown(f"- 总页数: **{meta['total_pages']}**")
        st.markdown(f"- 含表格页: **{meta['pages_with_tables']}**")
        st.markdown(f"- 结构化条目: **{len(st.session_state.parsed_result['structured_entries'])}**")
        st.markdown(f"- Chunks: **{st.session_state.retriever.collection.count()}**")

    st.markdown("---")
    st.markdown("### 🧪 预设问题")
    preset_questions = [
        "GB/T 1568-2008 标准的全称是什么？",
        "这个标准的适用范围是什么？",
        "标准中引用了哪些规范性文件？",
        "材料硬度要求中，HBW、HRC、HV 分别代表什么",
        "表格中的尺寸公差是多少？（请列出表格数据）",
        "标准中规定了哪些检验方法？",
        "这个标准关于飞机发动机制造有哪些要求？",
    ]
    for q in preset_questions:
        if st.button(f"📝 {q[:50]}...", key=f"preset_{q[:10]}"):
            st.session_state.current_question = q

    st.markdown("---")
    st.caption(f"Powered by DeepSeek | Agent v0.1.0")


# ─── 主界面 ────────────────────────────────────────────────────
st.title("🏦 保险文档智能问答 Agent")

# 标签页
tab1, tab2, tab3 = st.tabs(["💬 问答", "📋 文档解析结果", "📊 历史记录"])

# ─── Tab 1: 问答 ──────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_area(
            "请输入您的问题",
            value=st.session_state.get("current_question", ""),
            height=80,
            placeholder="例如：标准中的公差要求是什么？...",
        )

    with col2:
        top_k = st.slider("检索数量", 3, 10, 5)

    if st.button("🔍 提问", type="primary", disabled=not st.session_state.initialized):
        if not question.strip():
            st.warning("请输入问题")
        else:
            with st.spinner("正在分析..."):
                result = st.session_state.agent.answer(question, top_k)
                st.session_state.question_history.append(result)

            # 显示答案
            st.markdown("### 📝 回答")
            st.markdown(result["answer"])

            # 置信度标签
            conf = result["confidence"]
            conf_color = {
                "high": "green",
                "medium": "orange",
                "low": "red",
                "no_answer": "gray",
            }
            conf_score = result["confidence_score"]

            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
            with col_meta1:
                st.metric("置信度", f":{conf_color.get(conf, 'gray')}[{conf}]", delta=f"{conf_score:.0%}")
            with col_meta2:
                timing = result["timing"]
                st.metric("总耗时", f"{timing['total']:.2f}s")
            with col_meta3:
                st.metric("幻觉风险", "⚠️ 是" if result["is_hallucination_risk"] else "✅ 否")
            with col_meta4:
                st.metric("检索来源", f"{len(result['sources'])} 条")

            if result.get("refusal_reason"):
                st.info(f"📌 拒答原因: {result['refusal_reason']}")

            # 推理过程
            with st.expander("🧠 推理过程"):
                st.text(result["reasoning"])

            # 来源引用
            with st.expander("📚 来源引用"):
                for i, src in enumerate(result["sources"]):
                    st.markdown(f"**来源 {i+1}** (页码 {src['page']}, 类型: {src['source_type']}, 相关度: {src['relevance_score']:.3f})")
                    st.text(src["content_preview"])
                    st.markdown("---")

            # 耗时分析
            with st.expander("⏱️ 耗时分析"):
                st.json(result["timing"])

# ─── Tab 2: 文档解析结果 ──────────────────────────────────────
with tab2:
    if st.session_state.initialized:
        st.markdown("### 📄 解析后的文档结构")

        # 基本信息
        meta = st.session_state.parsed_result["metadata"]
        st.json(meta)

        # 各页面概览
        st.markdown("### 📑 页面概览")
        for page in st.session_state.parsed_result["pages"]:
            with st.expander(
                f"第 {page['page_num']} 页 "
                f"(OCR置信度: {page['ocr_confidence_avg']:.0f}%, "
                f"低置信词: {page['low_confidence_count']})"
            ):
                st.text(page["text"][:1000])

        # 表格展示
        st.markdown("### 📊 提取的表格")
        table_count = 0
        for page in st.session_state.parsed_result["pages"]:
            for tbl in page.get("tables", []):
                table_count += 1
                with st.expander(f"表格 {table_count} (第{page['page_num']}页, {tbl['rows']}行×{tbl['cols']}列)"):
                    if tbl["data"]:
                        st.table(tbl["data"])

        if table_count == 0:
            st.info("未提取到表格")
    else:
        st.info("请先初始化系统")

# ─── Tab 3: 历史记录 ──────────────────────────────────────────
with tab3:
    if st.session_state.question_history:
        for i, record in enumerate(reversed(st.session_state.question_history)):
            with st.expander(f"Q{i+1}: {record['question'][:80]}... ({record['confidence']})"):
                st.markdown(f"**Q:** {record['question']}")
                st.markdown(f"**A:** {record['answer']}")
                st.caption(f"置信度: {record['confidence']} ({record['confidence_score']:.0%}) | 耗时: {record['timing']['total']:.2f}s")
    else:
        st.info("暂无问答记录")

# ─── 页脚 ──────────────────────────────────────────────────────
st.markdown("---")
st.caption("保险文档智能问答 Agent | 基于 RAG + DeepSeek | v0.1.0")
