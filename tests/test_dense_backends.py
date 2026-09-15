"""Chroma 向量库：相关度排序、复用与嵌入维度切换自动整库重建。"""

import pytest

pytest.importorskip("chromadb")

from retail_assistant.models import Document  # noqa: E402
from retail_assistant.retrieval.dense import VectorStore  # noqa: E402

from fakes import HashingEmbedder  # noqa: E402


def _docs():
    return [
        Document(id="a", page_content="牛奶 早餐奶 乳制品", metadata={}),
        Document(id="b", page_content="抽纸 洗手液 日用品", metadata={}),
    ]


def test_chroma_ranks_relevant_first(tmp_path):
    docs = _docs()
    idx = VectorStore(HashingEmbedder(64), str(tmp_path / "chroma")).load_or_build(
        str(tmp_path), docs
    )
    assert [d.id for d, _ in idx.search("牛奶", 2)] == ["a", "b"]
    assert [d.id for d, _ in idx.search("抽纸", 2)] == ["b", "a"]


def test_chroma_load_or_build_reuses_aligned_collection(tmp_path):
    docs = _docs()
    persist = str(tmp_path / "chroma")
    VectorStore(HashingEmbedder(64), persist).load_or_build(str(tmp_path), docs)
    # doc_id 与维度都对齐：直接复用，重建后 search 结果应一致
    idx = VectorStore(HashingEmbedder(64), persist).load_or_build(str(tmp_path), docs)
    assert [d.id for d, _ in idx.search("牛奶", 2)] == ["a", "b"]


def test_chroma_rebuilds_on_embedding_dim_change(tmp_path):
    docs = _docs()
    persist = str(tmp_path / "chroma")
    VectorStore(HashingEmbedder(64), persist).load_or_build(str(tmp_path), docs)
    # 切换嵌入维度（模拟换 embedding 模型）：collection 应整库重建而非报错
    idx = VectorStore(HashingEmbedder(32), persist).load_or_build(str(tmp_path), docs)
    hits = idx.search("牛奶", 2)
    assert [d.id for d, _ in hits] == ["a", "b"]
