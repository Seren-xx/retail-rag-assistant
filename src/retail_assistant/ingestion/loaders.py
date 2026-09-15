"""多格式知识源加载：txt/md/csv/json/jsonl 原生，pdf/docx/xlsx 可选；表格统一拍平为"字段：值"文本。"""

import csv
import json
import re
import sys
from pathlib import Path

from ..models import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json", ".jsonl", ".pdf", ".docx", ".xlsx"}


class UnsupportedFormatError(ValueError):
    pass


def _meta(path: Path, kind: str, **extra) -> dict:
    metadata = {
        "source": path.name,
        "kind": kind,
        "balance_key": path.name,
    }
    metadata.update(extra)
    return metadata


def _table_lines(rows: list[list[str]]) -> list[str]:
    """统一表格转结构化文本：两列表出"字段：值"，多列表按表头配对，其余" | "拼接。"""
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    ncols = len(rows[0])
    if ncols == 2:
        return [f"{r[0]}：{r[1]}" for r in rows if r[0] or r[1]]
    if ncols >= 3:
        header = rows[0]
        lines = []
        for r in rows[1:]:
            pairs = [f"{h}：{v}" for h, v in zip(header, r) if h or v]
            if pairs:
                lines.append("；".join(pairs))
        return lines
    return [" | ".join(r) for r in rows]


_MD_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


def _md_table_lines(text: str) -> list[str]:
    """解析 Markdown 管道表格，转结构化文本；其余行原样保留。"""
    lines: list[str] = []
    buf: list[list[str]] = []

    def flush():
        if buf:
            lines.extend(_table_lines(buf))
            buf.clear()

    for raw in text.splitlines():
        m = _MD_TABLE_ROW.match(raw)
        # 分隔行（| --- | :---: |）跳过，不当作数据
        if m and not set(m.group(1)) <= set("-: "):
            buf.append([c.strip() for c in m.group(1).split("|")])
        else:
            flush()
            lines.append(raw)
    flush()
    return lines


def load_text(path: Path) -> list[Document]:
    """TXT / Markdown：整篇一个文档；Markdown 管道表格自动转结构化文本。"""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    lines = _md_table_lines(text) if path.suffix.lower() == ".md" else text.splitlines()
    return [Document(id=f"kb:{path.stem}", page_content="\n".join(lines), metadata=_meta(path, "knowledge"))]


def load_csv(path: Path, id_column: str | None = None) -> list[Document]:
    """CSV：每行一个结构化文档，文本为"列: 值"拼接，数值列写入 metadata 供硬过滤。"""
    docs = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row = {k: (v or "").strip() for k, v in row.items()}
            row_id = row.get(id_column) if id_column else None
            doc_id = f"product:{row_id}" if row_id else f"row:{path.stem}:{i}"
            text = "；".join(f"{k}：{v}" for k, v in row.items() if v)
            metadata = _meta(path, "product", balance_key=row_id or doc_id)
            for key in ("price", "stock"):
                if row.get(key):
                    try:
                        metadata[key] = float(row[key]) if key == "price" else int(row[key])
                    except ValueError:
                        pass
            for key in ("name", "brand", "category", "unit"):
                if row.get(key):
                    metadata[key] = row[key]
            metadata["volatile"] = True  # 价格/库存会变动
            docs.append(Document(id=doc_id, page_content=text, metadata=metadata))
    return docs


def load_jsonl(path: Path, id_field: str = "id") -> list[Document]:
    """JSONL：每行一个 JSON 对象，取 id 字段作为 doc_id。"""
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_id = obj.get(id_field) or f"line:{path.stem}:{i}"
            text = "；".join(f"{k}：{v}" for k, v in obj.items() if isinstance(v, str))
            metadata = _meta(path, "promotion", balance_key=row_id)
            for key in ("start_date", "end_date", "category", "store", "title", "discount"):
                if obj.get(key):
                    metadata[key] = obj[key]
            metadata["volatile"] = True  # 促销有生效时间
            docs.append(Document(id=str(row_id), page_content=text, metadata=metadata))
    return docs


def load_json(path: Path, id_field: str = "id") -> list[Document]:
    """JSON 数组：与 JSONL 同规则。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]
    docs = []
    for i, obj in enumerate(data):
        row_id = obj.get(id_field) if isinstance(obj, dict) else None
        text = "；".join(f"{k}：{v}" for k, v in obj.items() if isinstance(v, str))
        docs.append(
            Document(
                id=str(row_id or f"item:{path.stem}:{i}"),
                page_content=text,
                metadata=_meta(path, "knowledge", balance_key=row_id or f"item:{i}"),
            )
        )
    return docs


def load_pdf(path: Path) -> list[Document]:
    """PDF：优先 pdfplumber（文本 + 表格），未装时退回 pypdf（仅文本）。"""
    try:
        import pdfplumber
    except ImportError:
        return _load_pdf_text_only(path)
    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
            for table in page.extract_tables():
                rows = [[("" if c is None else str(c).strip()) for c in row] for row in table]
                parts.extend(_table_lines(rows))
    if not parts:
        return []
    return [Document(id=f"kb:{path.stem}", page_content="\n".join(parts), metadata=_meta(path, "knowledge"))]


def _load_pdf_text_only(path: Path) -> list[Document]:
    """pypdf 兜底：只抽文本，表格不保证还原（正常应安装 pdfplumber）。"""
    try:
        from pypdf import PdfReader

        print("warning: 未安装 pdfplumber，PDF 表格不抽取（仅文本）：pip install -e '.[ingest-pdf]'", file=sys.stderr)
    except ImportError as e:
        raise UnsupportedFormatError(
            f"解析 {path.name} 需要 pdfplumber（含表格抽取）：pip install -e \".[ingest-pdf]\""
        ) from e
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        return []
    return [Document(id=f"kb:{path.stem}", page_content=text, metadata=_meta(path, "knowledge"))]


def load_docx(path: Path) -> list[Document]:
    """DOCX：需 python-docx；段落正文 + 表格转结构化文本一起进入切分索引。"""
    try:
        import docx
    except ImportError as e:
        raise UnsupportedFormatError(
            f"解析 {path.name} 需要 python-docx：pip install -e \".[ingest-docx]\""
        ) from e
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        rows = [[c.text.strip() for c in row.cells] for row in table.rows]
        parts.extend(_table_lines(rows))
    if not parts:
        return []
    return [Document(id=f"kb:{path.stem}", page_content="\n".join(parts), metadata=_meta(path, "knowledge"))]


def load_xlsx(path: Path) -> list[Document]:
    """XLSX：需 openpyxl；每 sheet 一个文档，行数据经 _table_lines 转结构化文本。"""
    try:
        import openpyxl
    except ImportError as e:
        raise UnsupportedFormatError(
            f"解析 {path.name} 需要 openpyxl：pip install -e \".[ingest-xlsx]\""
        ) from e
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    docs = []
    for ws in wb.worksheets:
        rows = [
            [("" if c is None else str(c).strip()) for c in row]
            for row in ws.iter_rows(values_only=True)
        ]
        lines = _table_lines(rows)
        if not lines:
            continue
        doc_id = f"kb:{path.stem}:{ws.title}" if len(wb.worksheets) > 1 else f"kb:{path.stem}"
        docs.append(
            Document(
                id=doc_id,
                page_content="\n".join(lines),
                metadata=_meta(path, "knowledge", sheet=ws.title),
            )
        )
    wb.close()
    return docs


def load_any(path: str | Path, csv_id_column: str | None = "product_id") -> list[Document]:
    """按扩展名分派到对应 loader。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            f"暂不支持的格式 {suffix}，当前支持：{sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix in (".txt", ".md"):
        return load_text(path)
    if suffix == ".csv":
        return load_csv(path, id_column=csv_id_column)
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".xlsx":
        return load_xlsx(path)
    return load_docx(path)
