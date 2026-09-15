"""多格式 loaders：txt/md/csv/jsonl 分派与 doc_id 稳定性。"""

import json
from pathlib import Path

import pytest

from retail_assistant.ingestion import loaders


def test_load_md(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("# 政策\n支持退货。", encoding="utf-8")
    docs = loaders.load_text(p)
    assert docs[0].id == "kb:policy"
    assert docs[0].metadata["kind"] == "knowledge"


def test_load_csv_generates_stable_ids(tmp_path):
    p = tmp_path / "products.csv"
    p.write_text(
        "product_id,name,brand,category,price,stock,unit,description,keywords\n"
        "S001,测试牛奶,测试牌,乳制品,10.0,5,箱,好喝。,牛奶\n",
        encoding="utf-8",
    )
    docs = loaders.load_csv(p, id_column="product_id")
    assert docs[0].id == "product:S001"
    assert docs[0].metadata["price"] == 10.0
    assert docs[0].metadata["stock"] == 5
    assert docs[0].metadata["volatile"] is True


def test_load_jsonl(tmp_path):
    p = tmp_path / "promotions.jsonl"
    p.write_text(
        '{"id": "promo:1", "title": "满减", "content": "牛奶满100减20"}\n',
        encoding="utf-8",
    )
    docs = loaders.load_jsonl(p)
    assert docs[0].id == "promo:1"
    assert "满减" in docs[0].page_content


def test_load_json_array(tmp_path):
    p = tmp_path / "faq.json"
    p.write_text(json.dumps([{"id": "faq:1", "q": "退货", "a": "7天"}], ensure_ascii=False), encoding="utf-8")
    docs = loaders.load_json(p)
    assert docs[0].id == "faq:1"


def test_unsupported_format_raises(tmp_path):
    p = tmp_path / "data.xyz"
    p.write_bytes(b"x")
    with pytest.raises(loaders.UnsupportedFormatError):
        loaders.load_any(p)


def test_load_any_dispatch(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("纯文本知识", encoding="utf-8")
    docs = loaders.load_any(p)
    assert docs[0].id == "kb:note"


# ---------- Markdown 管道表格抽取 ----------

def test_load_md_extracts_pipe_table(tmp_path):
    p = tmp_path / "spec.md"
    p.write_text(
        "# 退货政策\n\n7 天内可退货。\n\n"
        "| 型号 | 价格 | 库存 |\n"
        "| --- | --- | --- |\n"
        "| S001 | 49.9 | 200 |\n"
        "| S002 | 39.9 | 150 |\n",
        encoding="utf-8",
    )
    text = loaders.load_text(p)[0].page_content
    assert "# 退货政策" in text                      # 正文保留
    assert "7 天内可退货。" in text
    assert "| 型号 | 价格 |" not in text             # 管道表格已转结构化文本
    assert "型号：S001；价格：49.9；库存：200" in text  # 表头配对
    assert "型号：S002；价格：39.9；库存：150" in text


def test_load_md_extracts_two_column_table(tmp_path):
    p = tmp_path / "fields.md"
    p.write_text(
        "| 保质期 | 6个月 |\n| 产地 | 内蒙古 |\n",
        encoding="utf-8",
    )
    text = loaders.load_text(p)[0].page_content
    assert "保质期：6个月" in text
    assert "产地：内蒙古" in text


# ---------- XLSX 表格抽取 ----------

def _make_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "库存"
    for row in (["型号", "价格", "库存"], ["S001", "49.9", "200"], ["S002", "39.9", "150"]):
        ws.append(row)
    p = tmp_path / "stock.xlsx"
    wb.save(str(p))
    return p


def test_load_xlsx_extracts_sheet_table(tmp_path):
    p = _make_xlsx(tmp_path)
    docs = loaders.load_xlsx(p)
    assert len(docs) == 1
    assert docs[0].id == "kb:stock"              # 单 sheet 不带工作表名
    assert "型号：S001；价格：49.9；库存：200" in docs[0].page_content
    assert docs[0].metadata["sheet"] == "库存"


def test_load_xlsx_multi_sheet_separate_docs(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "规格"
    ws1.append(["保质期", "6个月"])
    ws2 = wb.create_sheet("门店")
    ws2.append(["门店", "地址"])
    ws2.append(["一店", "北京路"])
    p = tmp_path / "multi.xlsx"
    wb.save(str(p))
    docs = loaders.load_xlsx(p)
    assert [d.id for d in docs] == ["kb:multi:规格", "kb:multi:门店"]  # 多 sheet 各自成文档


def test_load_any_dispatches_xlsx(tmp_path):
    p = _make_xlsx(tmp_path)
    docs = loaders.load_any(p)
    assert docs[0].id == "kb:stock"


# ---------- DOCX 表格抽取 ----------

def _make_docx(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("产品规格说明")
    # 两列表：字段/值
    t1 = doc.add_table(rows=3, cols=2)
    for i, (k, v) in enumerate([("保质期", "6个月"), ("产地", "内蒙古"), ("净含量", "250ml")]):
        t1.rows[i].cells[0].text = k
        t1.rows[i].cells[1].text = v
    # 多列表：首行表头
    t2 = doc.add_table(rows=2, cols=3)
    for i, h in enumerate(["型号", "价格", "库存"]):
        t2.rows[0].cells[i].text = h
    for i, v in enumerate(["S001", "49.9", "200"]):
        t2.rows[1].cells[i].text = v
    p = tmp_path / "spec.docx"
    doc.save(str(p))
    return p


def test_load_docx_extracts_two_column_table(tmp_path):
    p = _make_docx(tmp_path)
    text = loaders.load_docx(p)[0].page_content
    assert "产品规格说明" in text          # 正文保留
    assert "保质期：6个月" in text         # 两列表 → "字段：值"
    assert "净含量：250ml" in text


def test_load_docx_extracts_multi_column_table_by_header(tmp_path):
    p = _make_docx(tmp_path)
    text = loaders.load_docx(p)[0].page_content
    assert "型号：S001；价格：49.9；库存：200" in text  # 表头配对
