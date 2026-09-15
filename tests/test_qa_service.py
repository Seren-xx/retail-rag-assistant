"""QAService 端到端（测试替身）：数值确定性回答、检索证据、拒答。"""

from retail_assistant.application.qa_service import QAService
from retail_assistant.application.session import RedisSessionStore
from retail_assistant.config import load_settings

from fakes import FakeRedis, HashingEmbedder, NoopReranker  # noqa: E402

FAKE_EMBED_DIM = 256


def _qa(tmp_path):
    s = load_settings(
        data_dir="data",
        index_dir=str(tmp_path / "index"),
        today="2026-09-11",
        dashscope_api_key="",
    )
    return QAService(
        settings=s,
        embedder=HashingEmbedder(FAKE_EMBED_DIM),
        reranker=NoopReranker(),
        session_store=RedisSessionStore(client=FakeRedis()),
        force_sync=True,
    )


def test_numeric_answer_is_deterministic(tmp_path):
    qa = _qa(tmp_path)
    out = qa.answer("S001多少钱")
    assert out["route"] == "numeric"
    assert out["decision"] == "ok"
    assert "49.9" in out["answer"]  # 数字来自 CSV，不经 LLM


def test_semantic_returns_evidence(tmp_path):
    qa = _qa(tmp_path)
    out = qa.answer("怎么开发票")
    assert out["decision"] == "ok"
    assert any(e["doc_id"].startswith("kb:") for e in out["evidence"])


def test_no_answer_query_is_refused(tmp_path):
    qa = _qa(tmp_path)
    out = qa.answer("你们和山姆超市什么关系")
    assert out["decision"] != "ok"
    assert "人工客服" in out["answer"] or "不足以" in out["answer"]


def test_retrieve_reports_filters(tmp_path):
    qa = _qa(tmp_path)
    res = qa.retrieve("50元以下的乳制品有哪些")
    assert res.filters["category"] == "乳制品"
    assert res.filters["price_max"] == 50.0
    assert any(e["doc_id"] == "product:S003" for e in res.evidence)


def test_llm_filter_extractor_breaks_after_repeated_failures(qa):
    """LLM 约束抽取连续失败 2 次熔断：后续查询纯规则，不再调用模型。"""
    calls = {"n": 0}

    def failing_extractor(query):
        calls["n"] += 1
        return None  # 模拟 LLM 调用失败

    qa.filter_extractor = failing_extractor
    qa._extract_filters("牛奶推荐")
    qa._extract_filters("牛奶推荐")
    assert qa._extract_disabled
    qa._extract_filters("牛奶推荐")
    assert calls["n"] == 2  # 熔断后不再调用
