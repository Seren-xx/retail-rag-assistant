"""质量门禁与上下文组装。"""

from langchain_core.documents import Document

from retail_assistant import config
from retail_assistant.retrieval.context import build_context
from retail_assistant.retrieval.quality import evaluate_quality


def _doc(doc_id, text, **meta):
    return Document(id=doc_id, page_content=text, metadata=meta)


def test_no_candidate():
    r = evaluate_quality("牛奶", [], [], [])
    assert r.decision == "no_candidate"


def test_gibberish_rejected_by_word_score():
    doc = _doc("a", "蒙牛纯牛奶 早餐奶")
    # 真实行为：BM25 对无命中的查询返回空 → 双路一致度为 0
    r = evaluate_quality("手机维修", [], [(doc, 0.9)], [(doc, 0.0)])
    assert r.decision == "low_retrieval_confidence"
    assert r.coverage == 0.0


def test_full_coverage_short_query_passes_word_gate():
    doc = _doc("a", "发票开具说明：电子发票30天内凭小票编码申请。")
    r = evaluate_quality("怎么开发票", [(doc, 1.0)], [(doc, 1.0)], [(doc, 0.02)])
    assert r.word_score >= config.word_score_floor


def test_time_mismatch_for_wrong_month_promo():
    doc = _doc(
        "promo:x",
        "十月囤货节",
        kind="promotion",
        start_date="2026-10-01",
        end_date="2026-10-08",
    )
    r = evaluate_quality(
        "9月有什么活动",
        [(doc, 1.0)],
        [(doc, 1.0)],
        [(doc, 0.5)],
        today=__import__("datetime").date(2026, 9, 11),
    )
    assert r.decision == "time_mismatch"


def test_context_source_balance_and_budget():
    docs = [
        (_doc(f"a{i}", "长文本内容" * 30, source="srcA", balance_key="srcA"), 1.0)
        for i in range(4)
    ]
    docs.append((_doc("b", "短文档", source="srcB", balance_key="srcB"), 0.5))
    context, used = build_context(docs, max_chars=500, max_per_source=2)
    assert len([d for d in used if d.metadata["balance_key"] == "srcA"]) == 2
    assert any(d.id == "b" for d in used)
    assert "[b]" in context


def test_context_char_budget():
    docs = [(_doc(f"d{i}", "x" * 200, source="s", balance_key=f"s{i}"), 1.0) for i in range(10)]
    context, used = build_context(docs, max_chars=500, max_per_source=10)
    assert len(context) <= 500 + 30  # 预算 + 单块超限容忍
