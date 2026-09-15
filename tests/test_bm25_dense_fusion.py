"""BM25 / Dense / RRF：召回与融合的基础行为。"""

import pytest

from retail_assistant.models import Document
from retail_assistant.retrieval.bm25 import BM25Index
from retail_assistant.retrieval.dense import VectorStore
from retail_assistant.retrieval.fusion import concat_fuse, rrf_fuse

from fakes import HashingEmbedder  # noqa: E402

pytest.importorskip("chromadb")

DOCS = [
    Document(id="a", page_content="蒙牛纯牛奶 早餐 奶制品", metadata={}),
    Document(id="b", page_content="洗衣液 清洁用品", metadata={}),
    Document(id="c", page_content="牛奶 巧克力 礼盒", metadata={}),
]


def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.build(DOCS)
    hits = idx.search("牛奶", top_k=2)
    assert hits[0][0].id in ("a", "c")
    assert all(score > 0 for _, score in hits)


def test_bm25_no_match_returns_empty():
    idx = BM25Index()
    idx.build(DOCS)
    assert idx.search("手机", top_k=3) == []


def test_dense_top1(tmp_path):
    idx = VectorStore(HashingEmbedder(64), str(tmp_path / "chroma"))
    idx.build(DOCS)
    hits = idx.search("牛奶", top_k=2)
    assert hits[0][0].id in ("a", "c")


def test_dense_load_or_build_roundtrip(tmp_path):
    persist = str(tmp_path / "chroma")
    idx = VectorStore(HashingEmbedder(64), persist).load_or_build(str(tmp_path), DOCS)
    rebuilt = VectorStore(HashingEmbedder(64), persist).load_or_build(str(tmp_path), DOCS)
    hits_a = idx.search("牛奶", top_k=1)
    hits_b = rebuilt.search("牛奶", top_k=1)
    assert hits_a[0][0].id == hits_b[0][0].id


def test_rrf_merges_by_doc_id():
    d1 = Document(id="a", page_content="doc a", metadata={})
    d2 = Document(id="a", page_content="doc a", metadata={})  # 同 doc_id 的两个对象
    list1 = [(d1, 0.9)]
    list2 = [(d2, 0.8)]
    fused = rrf_fuse([list1, list2], k=60)
    assert len(fused) == 1  # 合并成一个而不是两个
    assert abs(fused[0][1] - 2 / 61) < 1e-9


def test_concat_fuse_interleaves_and_dedupes():
    a = Document(id="a", page_content="a", metadata={})
    b = Document(id="b", page_content="b", metadata={})
    c = Document(id="c", page_content="c", metadata={})
    merged = concat_fuse([[(a, 1.0), (c, 0.5)], [(b, 1.0), (a, 0.9)]])
    ids = [d.id for d, _ in merged]
    assert ids.count("a") == 1
    assert ids[0] == "a" and ids[1] == "b"
