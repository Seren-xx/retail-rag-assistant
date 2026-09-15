"""意图路由：结构化数值查询（catalog 确定性回答）vs 语义检索（混合检索）双通路。"""

import re

# 数值查询关键词（沿用 legacy 并收窄）：促销/优惠/活动列举类查询不走 numeric——
# 其答案在促销文档里，应走语义检索（含时效门禁），catalog 证据不含促销文档。
NUMERIC_KEYWORDS = [
    "价格", "多少钱", "售价", "原价",
    "库存", "剩余", "销量", "数量",
    "重量", "规格", "容量", "尺寸", "大小",
]

# 精确数值模式（沿用 legacy）
NUMERIC_PATTERNS = [
    r"\d+元", r"\d+块", r"\d+\.\d+元",
    r"低于\d+", r"高于\d+", r"超过\d+",
    r"\d+折", r"\d+%",
    r"价格.*\d+", r"\d+.*价格",
]

# SKU 编号模式（S001 等）；中文与 \w 同类，不能用 \b 边界
SKU_PATTERN = re.compile(r"(?<![A-Za-z0-9])[Ss]\d{3}(?![A-Za-z0-9])")


class IntentRouter:
    def is_numeric_query(self, query: str) -> bool:
        has_keyword = any(kw in query for kw in NUMERIC_KEYWORDS)
        has_pattern = any(re.search(p, query) for p in NUMERIC_PATTERNS)
        has_sku = SKU_PATTERN.search(query) is not None
        return has_keyword or has_pattern or has_sku

    def extract_numeric_constraints(self, query: str) -> dict:
        constraints = {}
        m = re.search(r"(\d+(?:\.\d+)?)\s*元", query)
        if m:
            constraints["price"] = float(m.group(1))
        m = re.search(r"(\d+)\s*折", query)
        if m:
            constraints["discount"] = int(m.group(1)) / 10
        if any(kw in query for kw in ["低于", "小于", "不超过", "以下"]):
            constraints["operator"] = "lte"
        elif any(kw in query for kw in ["高于", "大于", "超过", "以上"]):
            constraints["operator"] = "gte"
        skus = SKU_PATTERN.findall(query)
        if skus:
            constraints["skus"] = [s.upper() for s in skus]
        return constraints

    def route(self, query: str) -> tuple[str, dict]:
        """返回 (route_type, constraints)：numeric 或 semantic。"""
        if self.is_numeric_query(query):
            return "numeric", self.extract_numeric_constraints(query)
        return "semantic", {}
