"""三级去重：MD5 / SimHash / 服务组合。"""

from retail_assistant.indexing.dedup import (
    DeduplicationService,
    MD5Deduplicator,
    SimHashDeduplicator,
)


def test_md5_exact_duplicate(tmp_path):
    d = MD5Deduplicator(str(tmp_path / "md5.index"))
    assert not d.is_duplicate("一段文本")
    d.add("一段文本")
    assert d.is_duplicate("一段文本")
    assert not d.is_duplicate("另一段文本")


def test_simhash_near_duplicate(tmp_path):
    d = SimHashDeduplicator(str(tmp_path / "sim.json"), threshold=3)
    d.add("超市商品退货政策说明", "a")
    assert d.is_duplicate("超市商品退货政策说明")
    assert not d.is_duplicate("完全不同的另一段内容文字")


def test_service_three_level(tmp_path):
    s = DeduplicationService(str(tmp_path))
    s.add("内容A", "chunk_a")
    assert s.is_duplicate("内容A")
    assert not s.is_duplicate("内容B")
