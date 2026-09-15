"""LLM 接入：未配置 Key 返回 None、JSON 解析容错（不发起真实网络调用）。"""

from retail_assistant.application.llm import (
    _parse_filter_json,
    make_filter_extractor,
    make_llm,
)
from retail_assistant.config import load_settings


def _settings(**kw):
    base = dict(
        dashscope_api_key="",
    )
    base.update(kw)
    return load_settings(**base)


def test_make_llm_returns_none_without_key():
    assert make_llm(_settings()) is None


def test_filter_extractor_returns_none_without_key():
    assert make_filter_extractor(_settings(llm_filters=True)) is None


def test_filter_extractor_disabled_by_flag():
    assert make_filter_extractor(_settings(llm_filters=False, dashscope_api_key="sk-test")) is None


def test_parse_filter_json_tolerates_fences():
    raw = '结果如下：\n```json\n{"brands": ["蒙牛"], "category": null}\n```'
    assert _parse_filter_json(raw) == {"brands": ["蒙牛"], "category": None}


def test_parse_filter_json_rejects_garbage():
    assert _parse_filter_json("这句话里没有 JSON") is None
    assert _parse_filter_json("") is None
