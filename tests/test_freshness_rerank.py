"""时效处理：过期过滤与新鲜度加成；Noop 重排。"""

import datetime as dt

from retail_assistant.models import Document
from retail_assistant.retrieval.freshness import apply_freshness_boost, filter_expired

from fakes import NoopReranker  # noqa: E402

TODAY = dt.date(2026, 9, 11)


def _promo(doc_id, end):
    return Document(
        id=doc_id,
        page_content="促销",
        metadata={"kind": "promotion", "end_date": end},
    )


def test_expired_promo_filtered():
    hits = [
        (_promo("old", "2026-08-31"), 0.9),
        (_promo("live", "2026-09-30"), 0.8),
    ]
    kept = filter_expired(hits, TODAY)
    assert [d.id for d, _ in kept] == ["live"]


def test_freshness_boost_orders_recent_first():
    old = Document(id="old", page_content="x", metadata={"volatile": True, "updated_at": "2026-06-01"})
    new = Document(id="new", page_content="x", metadata={"volatile": True, "updated_at": "2026-09-10"})
    adjusted = apply_freshness_boost([(old, 1.0), (new, 1.0)], TODAY, weight=0.05, window_days=90)
    assert adjusted[0][0].id == "new"
    # 加成幅度不超过 weight
    assert adjusted[0][1] <= 1.0 + 0.05 + 1e-9


def test_noop_reranker_truncates():
    hits = [(Document(id=str(i), page_content="t", metadata={}), 1.0) for i in range(10)]
    assert len(NoopReranker().rerank("q", hits, top_k=5)) == 5
