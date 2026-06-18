# 🏦 保险文档智能问答 Agent

基于 RAG 的扫描 PDF 智能问答系统，支持 OCR 解析、表格提取、混合检索、多阶段推理和置信度自校验。

## 🏗 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     Web UI (Streamlit) / CLI / API               │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│              智能决策层 (Agent/Reasoner)                          │
│  Query ─→ 检索 ─→ 置信度预判 ─→ 生成答案 ─→ 自校验 ─→ 拒答/输出 │
│                         DeepSeek Chat API                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│              知识检索层 (RAG)                                     │
│  ┌─────────────────┐  ┌──────────────────┐                       │
│  │ 语义检索         │  │ 关键词检索 (BM25) │                       │
│  │ Sentence-Transformers│                │                       │
│  │ ChromaDB (向量)   │  │ 内存索引          │                       │
│  └────────┬────────┘  └────────┬─────────┘                       │
│           └──────────┬─────────┘                                 │
│                      ▼                                            │
│              混合融合 (加权合并)                                    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│              解析层 (Parser)                                      │
│  PDF ─→ PyMuPDF 渲染图像 ─→ OpenCV 预处理 ─→ Tesseract OCR      │
│     ─→ pdfplumber 表格提取 ─→ 结构化 Markdown + JSON              │
│     ─→ OCR 置信度标记 (低置信词记录)                               │
└──────────────────────────────────────────────────────────────────┘
```

## 🚀 快速启动

### 环境要求

- Python 3.11+
- Tesseract OCR 5.x (需安装中文语言包 `chi_sim`)
- Windows / Linux / macOS

### 1. 安装 Tesseract

**Windows:**
```bash
winget install UB-Mannheim.TesseractOCR
# 然后下载中文语言包到 tessdata 目录
# https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata
```

### 2. 安装 Python 依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# 编辑 .env 文件
DEEPSEEK_API_KEY=sk-your-key-here
LLM_MODEL=deepseek-chat
```

### 4. 启动

```bash
# 命令行演示
python run.py demo

# 或 Web 界面
python run.py ui

# 或 API 服务
python run.py api
```

## 📂 项目结构

```
.
├── configs/
│   └── settings.yaml          # 全局配置（OCR/Embedding/LLM/Agent）
├── src/
│   ├── config.py              # 配置加载器
│   ├── parser/
│   │   └── pdf_parser.py      # 解析层：PDF → OCR → 结构化
│   ├── rag/
│   │   └── retriever.py       # RAG 层：混合检索
│   ├── agent/
│   │   └── reasoner.py        # Agent 层：多阶段推理 + 自校验
│   └── api/
│       ├── server.py          # FastAPI 后端
│       └── ui.py              # Streamlit 前端
├── tests/
│   ├── test_parser.py         # 解析层单元测试
│   └── test_agent.py          # Agent 层测试（5类边界场景）
├── data/                      # 向量数据库持久化
├── outputs/                   # 解析结果输出
├── logs/                      # 日志
├── run.py                     # 主入口
├── .env                       # 密钥配置
└── README.md
```

## 🧠 核心设计

### 1. 解析层 — PDF → 结构化

| 步骤 | 技术 | 说明 |
|------|------|------|
| 渲染 | PyMuPDF (fitz) | 300 DPI 高保真渲染 |
| 预处理 | OpenCV | 灰度化 → 去噪 → 自适应阈值 |
| OCR | Tesseract 5.x | 中英文混合识别 + 逐词置信度 |
| 表格 | pdfplumber | 独立表格检测通道，保留行列结构 |

**边界处理：**
- OCR 置信度 < 60% → 标记为 `low_confidence`，记录具体词汇
- 表格提取失败 → 降级为纯文本，不阻塞流程
- 空白页 → 记录警告，继续处理

### 2. RAG 层 — 混合检索

```
语义检索 (α=0.7)  +  关键词检索 (1-α=0.3) = 混合排序
     ↓                        ↓
 ChromaDB 向量            BM25 内存索引
 Sentence-Transformers     中文2-gram + 英文分词
```

每个 Chunk 携带 Metadata：页码、来源类型、OCR置信度、条款编号

### 3. Agent 层 — 四阶段推理

```
用户问题
  → [阶段1] 检索相关文档片段
  → [阶段2] 置信度预判 (规则 + LLM 双通道)
  → [阶段3] DeepSeek 生成答案
  → [阶段4] 自校验 (规则校验 + LLM 反幻觉校验)
  → 最终输出 (含置信度 + 来源引用)
```

**拒答机制：**
- 阶段2 检索相关度极低 → 直接拒答，"无相关依据"
- 阶段4 发现大量编造数据 → 标记低置信度/幻觉风险

### 4. 置信度评估体系

| 维度 | 方法 |
|------|------|
| 检索相关度 | ChromaDB cosine distance + BM25 score |
| 预判置信度 | Top-3 平均相关度 > 阈值 → 通过 |
| 内容忠实度 | 答案中的数字/编号是否在原文中出现 |
| LLM 自校验 | 独立 LLM 调用检查是否编造事实 |

## 🧪 测试策略

### 5 类边界测试用例

| # | 类型 | 测试问题示例 | 验证目标 |
|---|------|-------------|---------|
| 1 | 表格问题 | "材料硬度要求的具体数值？" | 表格被检索并引用 |
| 2 | 无答案问题 | "飞机发动机要求？" | 触发拒答/低置信度 |
| 3 | OCR 模糊问题 | 针对低置信度页面提问 | 回答中有不确定性标记 |
| 4 | 条款追溯 | "第3章规定了哪些技术要求？" | 条款编号被正确检索 |
| 5 | 对比推理 | "HBW 和 HRC 的区别？" | 多页信息综合 |

运行测试：
```bash
python run.py test
```

## 🎯 设计取舍 (Trade-offs)

| 决策 | 选择 | 替代方案 | 原因 |
|------|------|---------|------|
| PDF 渲染 | PyMuPDF | pdf2image+poppler | Windows 免安装 poppler |
| 表格提取 | pdfplumber | Camelot (需 Ghostscript) | 减少系统依赖 |
| 中文分词 | 2-gram | jieba/FoolNLTK | 减少依赖，BM25 场景够用 |
| 向量数据库 | ChromaDB | FAISS/Qdrant | 轻量、带 Metadata、持久化简单 |
| Embedding 模型 | MiniLM-L12-v2 | BGE-M3 | 轻量 (384维)，启动快 |
| LLM | DeepSeek Chat | GPT-4o | 中文能力强、性价比高 |
| BM25 | 自实现 | rank-bm25 库 | 减少依赖，逻辑透明 |

## 🔧 业务场景扩展方案

### 场景1: 海外保险合规文档（英文为主）
- 切换 OCR 语言：`ocr_lang: "eng"`
- 切换 Embedding：`all-MiniLM-L6-v2`
- 添加专业术语字典到 BM25 分词

### 场景2: 金融合同数值密集型
- 增强数字/百分比提取正则
- 添加数值范围交叉验证
- 表格对齐校验（行列一致性）

### 场景3: 多文档知识库
- ChromaDB 分区 (collection per document type)
- 添加文档来源筛选
- 跨文档引用追踪

### 场景4: 实时流式处理
- 异步 Agent 流水线
- 支持 Server-Sent Events (SSE)
- 边缘 OCR (GPU 加速)

## ⚠️ 已知限制

1. **Tesseract OCR 对扫描质量敏感** — 分辨率 < 150 DPI 的文档准确率显著下降
2. **表格检测依赖 pdfplumber 规则** — 复杂嵌套表格可能漏检
3. **BM25 是简易实现** — 生产环境建议用 Elasticsearch
4. **Embedding 模型仅 384 维** — 对领域特化文档可替换为 BGE 等更大模型
5. **自校验非 100% 可靠** — LLM 可能标记不准确，建议人工复核低置信度回答

## 📄 许可

本项目仅供学习和面试评估使用。
