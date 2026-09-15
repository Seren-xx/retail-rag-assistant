"""查询约束提取与同义词扩展：纯规则实现（对应 legacy/intent_router.py 的 LLM 抽取改为确定性规则）。"""

import re
from dataclasses import dataclass, field

# 品牌表（与 data/products.csv 对齐；目录加载后可再补充）
BRANDS = [
    "蒙牛", "伊利", "光明", "简爱", "安慕希", "农夫山泉", "可口可乐", "元气森林",
    "金龙鱼", "鲁花", "北大荒", "三只松鼠", "旺旺", "奥利奥", "良品铺子",
    "立白", "蓝月亮", "维达", "舒肤佳", "高洁丝", "双汇", "龙大", "妙可蓝多", "三全",
]

# 口语 → 规范品类（与 products.csv 的 category 对齐）
CATEGORY_SYNONYMS = {
    "乳制品": ["乳制品", "牛奶", "酸奶", "奶制品", "奶酪", "芝士"],
    "饮料": ["饮料", "茶饮", "可乐", "汽水", "气泡水", "果汁"],
    "粮油": ["粮油", "大米", "食用油", "米面"],
    "零食": ["零食", "饼干", "坚果", "糖果", "巧克力", "膨化"],
    "日用清洁": ["清洁", "洗涤", "洗衣", "纸巾", "抽纸", "洗手液", "日用品"],
    "个护": ["个护", "香皂", "肥皂", "卫生巾", "护理"],
    "肉制品": ["肉制品", "火腿肠", "香肠", "卤味"],
    "生鲜": ["生鲜", "鲜肉", "猪肉", "牛肉", "蔬菜"],
    "速冻": ["速冻", "水饺", "饺子", "汤圆"],
}

# 查询扩展：口语词 → 追加检索词（别名召回，不改写原查询）
QUERY_ALIASES = {
    "多少钱": ["价格", "售价"],
    "怎么卖": ["价格"],
    "打折": ["促销", "优惠"],
    "优惠": ["促销", "活动"],
    "退货": ["退款", "退换"],
    "开发票": ["发票", "开票"],
    "发票": ["开票"],
    "无糖": ["0糖", "无蔗糖"],
    "酸奶": ["乳酸菌"],
    "牛奶": ["乳制品"],
    "水饺": ["饺子"],
    "洗手液": ["抑菌"],
    "送货": ["配送"],
    "快递": ["配送"],
}


# 口语功能词：内容词统计时剔除（查询理解与质量门禁共用）
STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "吗", "呢", "吧", "嘛", "么", "怎",
    "请", "你", "您", "好", "们", "想", "要", "能", "会", "问", "下", "个",
    "什么", "哪些", "多少", "请问", "怎么", "怎样", "可以", "一下", "有没有",
    "还是", "以及", "如何", "推荐",
    # 价格区间/口语中的功能字
    "元", "以", "上", "下", "左", "右", "还", "就", "都", "很", "挺", "太",
    "更", "最", "卖", "牌子", "左右", "几块", "一块",
}


def content_tokens(text: str) -> list[str]:
    """英文按词、中文按字提取内容词，剔除口语功能词。"""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    tokens += re.findall(r"[\u4e00-\u9fff]", text)
    return [t for t in tokens if t not in STOPWORDS]


@dataclass
class Filters:
    """召回前硬过滤条件（仅作用于结构化商品文档）。"""

    brands: list[str] = field(default_factory=list)
    category: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    months: list[int] = field(default_factory=list)
    raw_query: str = ""


def extract_filters(query: str) -> Filters:
    f = Filters(raw_query=query)

    # 品牌：查询中出现的所有品牌（多品牌比较时不用于过滤，由调用方判断）
    f.brands = [b for b in BRANDS if b in query]

    # 品类
    for canonical, synonyms in CATEGORY_SYNONYMS.items():
        if any(s in query for s in synonyms):
            f.category = canonical
            break

    # 价格区间
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-到~和]\s*(\d+(?:\.\d+)?)\s*元", query)
    if m:
        f.price_min, f.price_max = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"(?:低于|小于|不超过|以下|以内)\s*(\d+(?:\.\d+)?)", query)
        if m:
            f.price_max = float(m.group(1))
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*元?\s*(?:以下|以内|以内)", query)
            if m:
                f.price_max = float(m.group(1))
            else:
                m = re.search(r"(?:高于|大于|超过|以上)\s*(\d+(?:\.\d+)?)", query)
                if m:
                    f.price_min = float(m.group(1))
                else:
                    m = re.search(r"(\d+(?:\.\d+)?)\s*元\s*(?:以上|往上)", query)
                    if m:
                        f.price_min = float(m.group(1))

    # 月份（时效匹配用）
    f.months = [int(x) for x in re.findall(r"(\d{1,2})月", query)]
    return f


def expand_query(query: str) -> str:
    """别名扩展：原查询在前，命中的同义词追加在后。"""
    extras: list[str] = []
    for word, synonyms in QUERY_ALIASES.items():
        if word in query:
            extras.extend(s for s in synonyms if s not in query)
    return query if not extras else query + " " + " ".join(extras)


# LLM 输出白名单清洗参数
_LLM_BRAND_MAX = 5
_PRICE_RANGE = (0.0, 100000.0)


def sanitize_llm_filters(raw) -> dict:
    """白名单清洗 LLM 抽取结果：品类/价格/月份逐项校验，其余字段一律丢弃防幻觉带偏。"""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}

    brands = raw.get("brands")
    if isinstance(brands, list):
        cleaned = [str(b).strip() for b in brands if str(b).strip()]
        if cleaned:
            out["brands"] = cleaned[:_LLM_BRAND_MAX]

    category = raw.get("category")
    if isinstance(category, str) and category.strip() in CATEGORY_SYNONYMS:
        out["category"] = category.strip()

    for key in ("price_min", "price_max"):
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if _PRICE_RANGE[0] <= float(v) <= _PRICE_RANGE[1]:
                out[key] = float(v)

    months = raw.get("months")
    if isinstance(months, list):
        valid = sorted({int(m) for m in months if isinstance(m, (int, float)) and 1 <= int(m) <= 12})
        if valid:
            out["months"] = valid
    return out


def merge_filters(rule: Filters, llm: dict) -> Filters:
    """LLM 约束与规则结果合并：列表取并集、标量规则优先（LLM 不得推翻确定性提取）。"""
    merged = Filters(
        brands=list(dict.fromkeys(rule.brands + llm.get("brands", []))),
        category=rule.category or llm.get("category"),
        price_min=rule.price_min if rule.price_min is not None else llm.get("price_min"),
        price_max=rule.price_max if rule.price_max is not None else llm.get("price_max"),
        months=rule.months or llm.get("months", []),
        raw_query=rule.raw_query,
    )
    return merged
