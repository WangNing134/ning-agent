"""
FastAPI 后端服务

提供 REST API 接口：
  - POST /parse       解析 PDF
  - POST /ask         问答
  - GET  /health      健康检查
  - GET  /docs        API 文档
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.config import get_config
from src.parser.pdf_parser import PDFParser
from src.rag.retriever import HybridRetriever
from src.agent.reasoner import DocumentAgent

# ─── 全局状态 ──────────────────────────────────────────────────
app = FastAPI(
    title="保险文档智能问答 Agent",
    description="基于 RAG 的扫描 PDF 智能问答系统",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 懒加载单例
_retriever: HybridRetriever | None = None
_agent: DocumentAgent | None = None
_parsed_result: dict | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    confidence_score: float
    sources: list[dict]
    reasoning: str
    is_hallucination_risk: bool
    refusal_reason: str | None
    timing: dict


class ParseResponse(BaseModel):
    status: str
    total_pages: int
    total_entries: int
    total_chunks: int | None
    message: str


# ─── 初始化 ────────────────────────────────────────────────────

def init_system(pdf_path: str | None = None):
    """初始化完整系统：解析 + 向量库 + Agent"""
    global _retriever, _agent, _parsed_result

    cfg = get_config()

    if pdf_path is None:
        pdf_path = str(Path(__file__).parent.parent.parent / "GBT1568-2008.pdf")

    logger.info("=" * 50)
    logger.info("初始化文档问答系统...")
    logger.info("=" * 50)

    # Step 1: 解析
    logger.info("Step 1/3: 解析 PDF...")
    parser = PDFParser()
    _parsed_result = parser.parse(pdf_path)
    logger.info(f"  完成: {_parsed_result['metadata']['total_pages']} 页")

    # Step 2: 构建知识库
    logger.info("Step 2/3: 构建知识库...")
    _retriever = HybridRetriever()
    n_chunks = _retriever.build_from_parsed(_parsed_result)
    logger.info(f"  完成: {n_chunks} chunks")

    # Step 3: 初始化 Agent
    logger.info("Step 3/3: 初始化 Agent...")
    _agent = DocumentAgent(retriever=_retriever)
    logger.info("  完成")

    logger.info("=" * 50)
    logger.info("系统就绪! 可以开始问答。")
    logger.info("=" * 50)


# ─── API 路由 ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent_ready": _agent is not None,
        "retriever_ready": _retriever is not None,
        "parsed_pages": _parsed_result["metadata"]["total_pages"] if _parsed_result else 0,
    }


@app.post("/parse")
async def parse_pdf():
    """重新解析 PDF"""
    try:
        init_system()
        return ParseResponse(
            status="ok",
            total_pages=_parsed_result["metadata"]["total_pages"],
            total_entries=len(_parsed_result["structured_entries"]),
            total_chunks=_retriever.collection.count() if _retriever else 0,
            message="PDF 解析完成",
        )
    except Exception as e:
        logger.error(f"解析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """问答接口"""
    if _agent is None:
        init_system()

    try:
        result = _agent.answer(req.question, req.top_k)
        return AskResponse(**result)
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/parsed")
async def get_parsed():
    """获取解析结果预览"""
    if _parsed_result is None:
        raise HTTPException(status_code=400, detail="请先解析 PDF")
    return {
        "metadata": _parsed_result["metadata"],
        "preview": _parsed_result.get("full_markdown", "")[:5000],
    }


# ─── 启动 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run(
        app,
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        log_level="info",
    )
