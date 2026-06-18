"""
解析层核心：扫描件 PDF → OCR → 结构化文档

处理流程：
  1. PyMuPDF 将 PDF 页面渲染为图像
  2. OpenCV 图像预处理（去噪、自适应阈值）
  3. Tesseract 逐页 OCR（含置信度信息）
  4. pdfplumber 提取表格结构
  5. 合并输出结构化 Markdown 文档

边界处理：
  - OCR 低置信度文本标记（低于阈值 → low_confidence 标记）
  - 表格提取失败时降级为纯文本
  - 空白页/图片页跳过
  - 多语言混排（中英文）
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pdfplumber
from loguru import logger

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config


class PDFParser:
    """扫描件 PDF 解析器：将扫描 PDF 转换为结构化文档"""

    def __init__(self, config: dict | None = None):
        cfg = get_config()
        self.settings = config or cfg["parser"]
        pytesseract.pytesseract.tesseract_cmd = self.settings["tesseract_exe"]
        self.ocr_lang = self.settings["ocr_lang"]
        self.confidence_threshold = self.settings["ocr_confidence_threshold"]
        self.render_dpi = self.settings["render_dpi"]
        self.table_extraction_enabled = self.settings["table_extraction_enabled"]
        self._global_entries: list[dict] = []

    def parse(self, pdf_path: str, output_dir: str = "outputs") -> dict:
        """
        完整解析 PDF 文件。

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录

        Returns:
            {
                "metadata": {...},
                "pages": [{"page_num": 1, "text": "...", "tables": [...], "ocr_confidence": 0.92, ...}],
                "full_markdown": "...",
                "structured_entries": [{"page":1, "type":"text|table|clause", "content":"...", "confidence":0.9}]
            }
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始解析 PDF: {pdf_path}")
        start_time = datetime.now()

        # Phase 1: 图像渲染 + OCR
        pages = self._ocr_pages(pdf_path)

        # Phase 2: 表格提取
        if self.table_extraction_enabled:
            pages = self._extract_tables(pdf_path, pages)

        # Phase 3: 结构化整理
        structured = self._structure_document(pages)

        # Phase 4: 生成 Markdown
        full_md = self._build_markdown(pages, structured)

        # 保存输出
        result = {
            "metadata": {
                "source": str(pdf_path),
                "total_pages": len(pages),
                "processed_at": datetime.now().isoformat(),
                "ocr_language": self.ocr_lang,
                "pages_with_tables": sum(1 for p in pages if p.get("tables")),
            },
            "pages": pages,
            "structured_entries": structured,
            "full_markdown": full_md,
        }

        # 保存为 JSON 和 Markdown
        stem = pdf_path.stem
        with open(output_dir / f"{stem}_parsed.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        md_path = output_dir / f"{stem}_parsed.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"解析完成: {len(pages)} 页, "
            f"{len(structured)} 个结构化条目, "
            f"耗时 {elapsed:.1f}s"
        )
        logger.info(f"输出: {md_path}")

        return result

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """图像预处理：灰度化 → 去噪 → 自适应阈值"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        if self.settings.get("preprocess_adaptive_threshold", True):
            processed = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed

    def _ocr_pages(self, pdf_path: Path) -> list[dict]:
        """逐页 OCR（使用行级检测 + 字级置信度）"""
        pages = []
        doc = fitz.open(str(pdf_path))

        for page_idx in range(len(doc)):
            try:
                page = doc[page_idx]
                # 渲染为图像
                mat = page.get_pixmap(dpi=self.render_dpi)
                img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
                img_np = np.array(img)

                # 预处理
                processed = self._preprocess_image(img_np)

                # Tesseract OCR — 行级输出 (level=4) 获取更好的文本流
                ocr_lines = pytesseract.image_to_data(
                    processed, lang=self.ocr_lang,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 6",  # 统一文本块模式
                )

                # 按行 (level=4/5, block_num + par_num + line_num) 分组
                lines_text, all_confidences = self._group_ocr_by_line(ocr_lines)

                full_text = "\n".join(lines_text)
                avg_conf = (
                    sum(all_confidences) / len(all_confidences)
                    if all_confidences else 0
                )

                # 记录低置信度词
                low_conf_words = []
                for i in range(len(ocr_lines["text"])):
                    t = ocr_lines["text"][i].strip()
                    c = int(ocr_lines["conf"][i]) if ocr_lines["conf"][i] != "-1" else 0
                    if t and 0 < c < self.confidence_threshold:
                        low_conf_words.append({"text": t, "confidence": c})

                page_data = {
                    "page_num": page_idx + 1,
                    "text": full_text,
                    "word_count": len(full_text),
                    "ocr_confidence_avg": round(avg_conf, 2),
                    "ocr_confidence_min": min(all_confidences) if all_confidences else 0,
                    "low_confidence_words": low_conf_words[:20],
                    "low_confidence_count": len(low_conf_words),
                    "tables": [],
                }

                pages.append(page_data)
                logger.debug(
                    f"  第{page_idx+1}页: {page_data['word_count']}字符, "
                    f"平均置信度 {avg_conf:.0f}%, "
                    f"低置信词 {len(low_conf_words)}个"
                )

            except Exception as e:
                logger.error(f"第 {page_idx+1} 页 OCR 失败: {e}")
                pages.append({
                    "page_num": page_idx + 1,
                    "text": f"[OCR 失败: {e}]",
                    "word_count": 0,
                    "ocr_confidence_avg": 0,
                    "ocr_confidence_min": 0,
                    "low_confidence_words": [],
                    "low_confidence_count": 0,
                    "tables": [],
                    "error": str(e),
                })

        doc.close()
        return pages

    @staticmethod
    def _group_ocr_by_line(ocr_data: dict) -> tuple[list[str], list[int]]:
        """
        将 Tesseract level=5 (word) 数据按行分组。

        Tesseract 字段：
          - level: 层级 (5=word)
          - block_num, par_num, line_num, word_num: 层级编号
          - text: 文本
          - conf: 置信度
        """
        lines: dict[tuple, list[str]] = {}
        confidences: list[int] = []

        for i in range(len(ocr_data["text"])):
            t = ocr_data["text"][i].strip()
            if not t:
                continue
            c = int(ocr_data["conf"][i]) if ocr_data["conf"][i] != "-1" else 0
            if c > 0:
                confidences.append(c)

            # 按 (block_num, par_num, line_num) 分组
            key = (
                ocr_data["block_num"][i],
                ocr_data["par_num"][i],
                ocr_data["line_num"][i],
            )
            if key not in lines:
                lines[key] = []
            lines[key].append(t)

        # 每行合并词 → 按 block/par/line 排序
        sorted_keys = sorted(lines.keys(), key=lambda k: (k[0], k[1], k[2]))
        lines_text = [" ".join(lines[k]) for k in sorted_keys]

        return lines_text, confidences

    def _extract_tables(self, pdf_path: Path, pages: list[dict]) -> list[dict]:
        """使用 pdfplumber + 文本模式匹配 双重策略提取表格"""

        # 策略1: pdfplumber (对文字型 PDF 有效)
        try:
            plumber_pdf = pdfplumber.open(str(pdf_path))
            for page_idx in range(len(plumber_pdf.pages)):
                plumber_page = plumber_pdf.pages[page_idx]
                detected_tables = plumber_page.extract_tables()
                if detected_tables:
                    tables_structured = self._clean_tables(detected_tables)
                    if tables_structured:
                        pages[page_idx]["tables"].extend(tables_structured)
                        logger.debug(f"  第{page_idx+1}页 (pdfplumber): {len(tables_structured)} 个表格")
            plumber_pdf.close()
        except Exception as e:
            logger.warning(f"pdfplumber 表格提取失败: {e}")

        # 策略2: 基于 OCR 文本的表格模式检测 (对扫描件有效)
        for page_idx in range(len(pages)):
            text = pages[page_idx]["text"]
            text_tables = self._detect_tables_from_text(text)
            if text_tables:
                existing_count = len(pages[page_idx].get("tables", []))
                for t in text_tables:
                    t["table_index"] = existing_count + 1
                    t["detection_method"] = "text_pattern"
                    existing_count += 1
                pages[page_idx]["tables"].extend(text_tables)
                logger.debug(f"  第{page_idx+1}页 (文本模式): {len(text_tables)} 个表格")

        return pages

    def _clean_tables(self, detected_tables: list) -> list[dict]:
        """清洗 pdfplumber 表格"""
        tables = []
        for tbl_idx, tbl in enumerate(detected_tables):
            cleaned = []
            for row in tbl:
                cleaned_row = [cell if cell is not None else "" for cell in row]
                cleaned.append(cleaned_row)
            if cleaned and any(any(cell for cell in row) for row in cleaned):
                tables.append({
                    "table_index": tbl_idx + 1,
                    "rows": len(cleaned),
                    "cols": len(cleaned[0]) if cleaned else 0,
                    "data": cleaned,
                    "header": cleaned[0] if cleaned else [],
                })
        return tables

    def _detect_tables_from_text(self, text: str) -> list[dict]:
        """
        从 OCR 文本中检测表格。

        启发式规则：
          1. 找连续2行以上具有相似列模式的行（按空格/制表符分割）
          2. 列数 ≥ 3 且各行列数一致
          3. 包含数字的列优先
        """
        lines = text.split("\n")
        if len(lines) < 3:
            return []

        tables = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # 尝试用多空格/制表符分割
            tokens = re.split(r'\s{2,}|\t+', line)
            tokens = [t.strip() for t in tokens if t.strip()]

            # 有效表格行：3列以上，且至少1列是数字
            has_number = any(re.search(r'\d', t) for t in tokens)
            if len(tokens) >= 3 and has_number:
                # 找后续连续行
                table_rows = [tokens]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        break
                    next_tokens = re.split(r'\s{2,}|\t+', next_line)
                    next_tokens = [t.strip() for t in next_tokens if t.strip()]
                    # 列数相同或 ±1
                    if abs(len(next_tokens) - len(tokens)) <= 1:
                        table_rows.append(next_tokens)
                        j += 1
                    else:
                        break

                if len(table_rows) >= 2:
                    # 统一列数到最大
                    max_cols = max(len(r) for r in table_rows)
                    padded_rows = []
                    for r in table_rows:
                        padded = r + [""] * (max_cols - len(r))
                        padded_rows.append(padded)

                    tables.append({
                        "table_index": 0,
                        "rows": len(padded_rows),
                        "cols": max_cols,
                        "data": padded_rows,
                        "header": padded_rows[0],
                        "detection_method": "text_pattern",
                    })
                    i = j
                    continue

            i += 1

        return tables

    def _structure_document(self, pages: list[dict]) -> list[dict]:
        """
        将页面文本结构化：
        - 识别条款编号（如 3.1、3.1.1）
        - 区分正文 / 条款 / 表格
        - 记录 OCR 置信度
        """
        entries = []
        clause_pattern = re.compile(
            r'^(\d+(?:\.\d+)*)\s+(.+)'  # 条款编号
        )

        for page in pages:
            page_num = page["page_num"]
            text = page["text"]

            # 先尝试按段落拆分
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for para in paragraphs:
                entry: dict[str, Any] = {
                    "page": page_num,
                    "content": para,
                    "confidence": page["ocr_confidence_avg"],
                }

                # 判断类型
                m = clause_pattern.match(para)
                if m:
                    entry["type"] = "clause"
                    entry["clause_number"] = m.group(1)
                    entry["clause_title"] = m.group(2)[:100]
                else:
                    entry["type"] = "text"

                # OCR 模糊标记
                if page["low_confidence_count"] > 0:
                    entry["ocr_quality"] = "low" if page["ocr_confidence_avg"] < self.confidence_threshold else "medium"

                entries.append(entry)

            # 表格条目
            for tbl in page.get("tables", []):
                entry = {
                    "page": page_num,
                    "type": "table",
                    "content": self._table_to_text(tbl),
                    "table_data": tbl["data"],
                    "table_header": tbl["header"],
                    "table_rows": tbl["rows"],
                    "table_cols": tbl["cols"],
                    "confidence": page["ocr_confidence_avg"],
                }
                entries.append(entry)

        self._global_entries = entries
        return entries

    @staticmethod
    def _table_to_text(tbl: dict) -> str:
        """将表格转为可读文本"""
        lines = []
        for row in tbl["data"]:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)

    def _build_markdown(self, pages: list[dict], structured: list[dict]) -> str:
        """生成完整 Markdown 文档"""
        md_lines = [
            f"# 扫描 PDF 解析结果",
            f"",
            f"**解析时间**: {datetime.now().isoformat()}",
            f"**总页数**: {len(pages)}",
            f"**OCR 引擎**: Tesseract ({self.ocr_lang})",
            f"**平均置信度**: {sum(p['ocr_confidence_avg'] for p in pages) / max(len(pages), 1):.1f}%",
            f"",
            "---",
            "",
        ]

        for page in pages:
            md_lines.append(f"## 第 {page['page_num']} 页")
            md_lines.append("")
            md_lines.append(
                f"> OCR 置信度: 平均 {page['ocr_confidence_avg']:.1f}%  |  "
                f"低置信词: {page['low_confidence_count']} 个"
            )
            md_lines.append("")

            # 正文
            md_lines.append(page["text"])
            md_lines.append("")

            # 表格
            for tbl in page.get("tables", []):
                md_lines.append(f"### 表格 {tbl['table_index']} (第{page['page_num']}页)")
                md_lines.append("")
                if tbl["data"]:
                    header = tbl["data"][0]
                    separator = ["---"] * len(header)
                    md_lines.append("| " + " | ".join(header) + " |")
                    md_lines.append("| " + " | ".join(separator) + " |")
                    for row in tbl["data"][1:]:
                        padded = (row + [""] * len(header))[:len(header)]
                        md_lines.append("| " + " | ".join(str(c) for c in padded) + " |")
                md_lines.append("")

            md_lines.append("---")
            md_lines.append("")

        return "\n".join(md_lines)
