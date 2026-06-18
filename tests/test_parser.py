"""
解析层单元测试

覆盖：
  1. PDF 页面渲染
  2. OCR 文本提取
  3. 表格提取
  4. 低置信度标记
  5. 结构化输出
"""

import sys
import json
import io
from pathlib import Path

# 修复 Windows GBK 编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.pdf_parser import PDFParser
from src.config import get_config


def test_pdf_parsing():
    """测试完整的 PDF 解析流程"""
    print("\n" + "=" * 60)
    print("测试 1: PDF 解析")
    print("=" * 60)

    pdf_path = Path(__file__).parent.parent / "GBT1568-2008.pdf"
    parser = PDFParser()
    result = parser.parse(str(pdf_path))

    # 断言
    assert "metadata" in result, "缺少 metadata"
    assert "pages" in result, "缺少 pages"
    assert "structured_entries" in result, "缺少 structured_entries"
    assert result["metadata"]["total_pages"] > 0, "没有解析出页面"

    print(f"✅ 解析成功: {result['metadata']['total_pages']} 页")

    # 每页有文本
    for page in result["pages"]:
        assert "text" in page, f"第{page['page_num']}页缺少文本"
        assert "ocr_confidence_avg" in page, f"第{page['page_num']}页缺少置信度"
        print(f"   第{page['page_num']}页: "
              f"{page['word_count']}字符, "
              f"OCR置信度 {page['ocr_confidence_avg']:.0f}%")

    return result


def test_ocr_quality(result: dict):
    """测试 OCR 质量标记"""
    print("\n" + "=" * 60)
    print("测试 2: OCR 质量标记")
    print("=" * 60)

    pages = result["pages"]
    total_low_conf = sum(p["low_confidence_count"] for p in pages)

    print(f"总低置信度词数: {total_low_conf}")

    for page in pages:
        if page["low_confidence_count"] > 0:
            print(f"   第{page['page_num']}页: {page['low_confidence_count']} 个低置信词")
            for w in page["low_confidence_words"][:5]:
                print(f"     '{w['text']}' (置信度: {w['confidence']})")

    # 记录低置信词 (不阻塞，只是信息)
    if total_low_conf > 0:
        print(f"⚠️ 存在 {total_low_conf} 个低置信度词，建议人工复核")
    else:
        print("✅ 所有文本置信度良好")

    return total_low_conf


def test_table_extraction(result: dict):
    """测试表格提取"""
    print("\n" + "=" * 60)
    print("测试 3: 表格提取")
    print("=" * 60)

    pages_with_tables = [
        p for p in result["pages"] if p.get("tables")
    ]

    print(f"提取到表格的页数: {len(pages_with_tables)}")

    for page in pages_with_tables:
        for tbl in page["tables"]:
            print(f"   第{page['page_num']}页 表格{tbl['table_index']}: "
                  f"{tbl['rows']}行×{tbl['cols']}列")
            print(f"     表头: {tbl['header']}")
            # 显示前3行
            for row in tbl["data"][:3]:
                print(f"     行: {row}")

    # 表格条目已加入结构化输出
    table_entries = [
        e for e in result["structured_entries"]
        if e["type"] == "table"
    ]
    print(f"结构化表格条目: {len(table_entries)}")

    return pages_with_tables


def test_structured_output(result: dict):
    """测试结构化输出完整性"""
    print("\n" + "=" * 60)
    print("测试 4: 结构化输出")
    print("=" * 60)

    entries = result["structured_entries"]
    print(f"结构化条目总数: {len(entries)}")

    type_counts = {}
    for e in entries:
        t = e["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, c in type_counts.items():
        print(f"   {t}: {c} 条")

    # 检查每个条目都有必需字段
    required_fields = ["page", "type", "content", "confidence"]
    for e in entries:
        for f in required_fields:
            assert f in e, f"条目缺少字段: {f}"

    print("✅ 所有条目字段完整")

    # 保存完整结构化输出到日志
    output_path = Path(__file__).parent.parent / "outputs" / "test_structured.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2, default=str)

    return entries


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█  解析层测试套件")
    print("█" * 60)

    try:
        result = test_pdf_parsing()
        test_ocr_quality(result)
        test_table_extraction(result)
        test_structured_output(result)

        print("\n" + "=" * 60)
        print("✅ 所有解析层测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
