"""指标计算与会话存储。"""

import json
import math

from fakes import FakeRedis
from retail_assistant.application.session import RedisSessionStore, make_session_store
from retail_assistant.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_hit():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
    assert recall_at_k(["a", "b", "x"], ["a", "c"], k=3) == 0.5
    assert hit_at_k(["x", "b"], ["b"], k=2) == 1.0
    assert hit_at_k(["x", "y"], ["b"], k=2) == 0.0


def test_mrr():
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_ndcg_binary():
    # 1 个相关文档排在第 1 位 → nDCG = 1
    assert ndcg_at_k(["a", "b"], ["a"], k=2) == 1.0
    # 排在第 2 位
    assert abs(ndcg_at_k(["b", "a"], ["a"], k=2) - (1 / math.log2(3)) / 1.0) < 1e-9


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.95) == 0.0
    assert percentile([10, 20], 0.95) == 19.5


# ---------- Redis 会话存储（client 注入，测试不依赖 redis 服务） ----------

def test_redis_session_roundtrip():
    store = RedisSessionStore(client=FakeRedis())
    store.add("u1", "user", "牛奶多少钱")
    store.add("u1", "assistant", "49.9元")
    history = store.get("u1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "49.9元"
    store.clear("u1")
    assert store.get("u1") == []


def test_redis_session_trims_to_max_messages():
    store = RedisSessionStore(client=FakeRedis(), max_messages=3)
    for i in range(5):
        store.add("u1", "user", f"m{i}")
    assert [m["content"] for m in store.get("u1")] == ["m2", "m3", "m4"]


def test_redis_session_corrupt_entry_skipped():
    client = FakeRedis()
    client.rpush(
        "retail:session:u1",
        "{broken",
        json.dumps({"role": "user", "content": "hi"}, ensure_ascii=False),
    )
    assert RedisSessionStore(client=client).get("u1") == [{"role": "user", "content": "hi"}]


def test_make_session_store_connects_redis(monkeypatch):
    import retail_assistant.application.session as session_mod

    monkeypatch.setattr(session_mod, "_connect_redis", lambda url: FakeRedis())
    store = session_mod.make_session_store("redis://test:6379/0")
    assert isinstance(store, RedisSessionStore)
    assert store.redis_url == "redis://test:6379/0"
