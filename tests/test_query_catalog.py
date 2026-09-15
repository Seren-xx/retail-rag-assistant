"""约束提取、别名扩展、意图路由与结构化目录。"""

import datetime as dt

from retail_assistant.knowledge.catalog import Catalog
from retail_assistant.query.constraints import (
    expand_query,
    extract_filters,
    merge_filters,
    sanitize_llm_filters,
)
from retail_assistant.query.router import IntentRouter


def test_extract_price_range():
    f = extract_filters("50元以下的牛奶")
    assert f.price_max == 50.0
    f = extract_filters("价格高于100元的油")
    assert f.price_min == 100.0
    f = extract_filters("30到60元的零食")
    assert (f.price_min, f.price_max) == (30.0, 60.0)


def test_extract_brand_and_multi_brand():
    f = extract_filters("蒙牛和伊利的牛奶哪个便宜")
    assert set(f.brands) == {"蒙牛", "伊利"}


def test_extract_category_synonym():
    assert extract_filters("牛奶推荐").category == "乳制品"
    assert extract_filters("抽纸有什么牌子").category == "日用清洁"


def test_month_extraction():
    assert extract_filters("9月有什么活动").months == [9]


def test_expand_query_appends_aliases():
    expanded = expand_query("牛奶多少钱")
    assert "价格" in expanded
    assert expanded.startswith("牛奶多少钱")


def test_router_routes_numeric_and_semantic():
    router = IntentRouter()
    route, constraints = router.route("S001多少钱")
    assert route == "numeric" and constraints.get("skus") == ["S001"]
    route, _ = router.route("生鲜能退货吗")
    assert route == "semantic"


def test_catalog_price_filter(settings):
    catalog = Catalog.load(settings)
    rows = catalog.filter_products(category="乳制品", price_max=50.0)
    ids = {r["product_id"] for r in rows}
    assert ids == {"S001", "S003", "S004", "S023"}


def test_catalog_active_promotions_respects_expiry(settings):
    catalog = Catalog.load(settings)
    active = catalog.active_promotions(dt.date(2026, 9, 11))
    ids = {p["id"] for p in active}
    assert "promo:2026-08-snack" not in ids  # 已过期
    assert "promo:2026-09-dairy" in ids
    assert "promo:2026-10-dairy" not in ids  # 未开始


def test_catalog_numeric_answer_contains_price(settings):
    catalog = Catalog.load(settings)
    filters = extract_filters("S001多少钱")
    text = catalog.answer_numeric(filters, dt.date(2026, 9, 11), skus=["S001"])
    assert text and "49.9" in text


# ---------- LLM 约束：白名单清洗与合并 ----------

def test_sanitize_llm_filters_drops_invalid_fields():
    cleaned = sanitize_llm_filters(
        {
            "brands": ["蒙牛", "", "  伊利 "],
            "category": " spaceship ",          # 非法品类 → 丢弃
            "price_min": -5,                     # 越界 → 丢弃
            "price_max": 50,
            "months": [0, 9, 13, "10"],          # 0/13 非法，字符串数字忽略
            "hack_field": "DROP ME",             # 白名单外 → 丢弃
        }
    )
    assert cleaned["brands"] == ["蒙牛", "伊利"]
    assert "category" not in cleaned
    assert "price_min" not in cleaned
    assert cleaned["price_max"] == 50.0
    assert cleaned["months"] == [9]
    assert "hack_field" not in cleaned


def test_sanitize_llm_filters_rejects_non_dict():
    assert sanitize_llm_filters(None) == {}
    assert sanitize_llm_filters("不是dict") == {}


def test_merge_filters_rule_wins_and_lists_union():
    rule = extract_filters("50元以下的蒙牛乳制品 9月")
    llm = sanitize_llm_filters(
        {"brands": ["伊利"], "category": "饮料", "price_max": 999, "months": [3]}
    )
    merged = merge_filters(rule, llm)
    assert set(merged.brands) == {"蒙牛", "伊利"}   # 列表取并集
    assert merged.category == "乳制品"              # 标量规则优先
    assert merged.price_max == 50.0                 # 规则 price_max 不被 LLM 推翻
    assert merged.months == [9]


def test_merge_filters_fills_missing_from_llm():
    rule = extract_filters("有什么推荐的")
    llm = sanitize_llm_filters({"category": "零食", "price_max": 30})
    merged = merge_filters(rule, llm)
    assert merged.category == "零食"
    assert merged.price_max == 30.0
