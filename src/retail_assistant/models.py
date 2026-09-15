"""统一数据模型：langchain Document + 文档库序列化。"""

from langchain_core.documents import Document

__all__ = ["Document", "doc_to_dict", "doc_from_dict"]


def doc_to_dict(doc: Document) -> dict:
    """文档 → docstore.jsonl 行。"""
    return {"doc_id": doc.id, "text": doc.page_content, "metadata": doc.metadata}


def doc_from_dict(data: dict) -> Document:
    """docstore.jsonl 行 → 文档（doc_id 稳定，重同步不变）。"""
    return Document(id=data["doc_id"], page_content=data["text"], metadata=data.get("metadata", {}))
