"""结构化商品目录：价格、库存、促销等事实数据走确定性查询（数字不经 LLM）。"""

import csv
import json
import os
from datetime import date

from ..query.constraints import Filters


class Catalog:
    def __init__(self, products: list[dict], promotions: list[dict]):
        self.products = products
        self.promotions = promotions
        self._by_id = {p.get("product_id"): p for p in products}

    @classmethod
    def load(cls, settings) -> "Catalog":
        products: list[dict] = []
        promotions: list[dict] = []

        csv_path = os.path.join(settings.data_dir, "products.csv")
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    row = {k: (v or "").strip() for k, v in row.items()}
                    for key in ("price", "stock"):
                        try:
                            row[key] = float(row[key]) if key == "price" else int(row[key])
                        except (KeyError, ValueError):
                            pass
                    products.append(row)

        promo_path = os.path.join(settings.data_dir, "promotions.jsonl")
        if os.path.exists(promo_path):
            with open(promo_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        promotions.append(json.loads(line))

        return cls(products, promotions)

    def get(self, product_id: str) -> dict | None:
        return self._by_id.get(product_id.upper())

    def filter_products(
        self,
        brands: list[str] | None = None,
        category: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        keyword: str | None = None,
    ) -> list[dict]:
        rows = self.products
        if brands:  # 多品牌取并集，避免比较类查询被错过滤成单一品牌
            rows = [r for r in rows if r.get("brand") in brands]
        if category:
            rows = [r for r in rows if r.get("category") == category]
        if price_min is not None:
            rows = [r for r in rows if isinstance(r.get("price"), (int, float)) and r["price"] >= price_min]
        if price_max is not None:
            rows = [r for r in rows if isinstance(r.get("price"), (int, float)) and r["price"] <= price_max]
        if keyword:
            rows = [r for r in rows if keyword in r.get("name", "") or keyword in r.get("keywords", "")]
        return rows

    def active_promotions(self, today: date, category: str | None = None) -> list[dict]:
        active = []
        for p in self.promotions:
            try:
                start = date.fromisoformat(p["start_date"][:10])
                end = date.fromisoformat(p["end_date"][:10])
            except (KeyError, ValueError):
                continue
            if not (start <= today <= end):
                continue
            if category and p.get("category") not in (category, "全品类"):
                continue
            active.append(p)
        return active

    def _narrow_by_query(self, rows: list[dict], query: str) -> list[dict]:
        """当品类/价格过滤后结果过多时，用查询内容词在名称与别名里收窄。

        例："花生巧克力多少钱" → 品类零食命中 4 款 → 收窄到名称含花生/巧/克/力的商品。
        """
        from ..query.constraints import content_tokens

        tokens = content_tokens(query)
        narrowed = [
            r for r in rows
            if any(
                t in f"{r.get('name', '')}{r.get('keywords', '')}"
                for t in tokens
            )
        ]
        return narrowed

    def format_product(self, row: dict) -> str:
        return (
            f"{row.get('product_id', '')} {row.get('name', '')}"
            f"（{row.get('brand', '')}）：售价 {row.get('price', '-')} 元，"
            f"库存 {row.get('stock', '-')} 件，规格 {row.get('unit', '-')}"
        )

    def answer_numeric(self, filters: Filters, today: date, skus: list[str] | None = None) -> str | None:
        """根据过滤条件生成确定性答案；无可匹配商品时返回 None（转语义通路）。"""
        if skus:
            rows = [r for r in (self.get(s) for s in skus) if r]
        else:
            rows = self.filter_products(
                brands=filters.brands,
                category=filters.category,
                price_min=filters.price_min,
                price_max=filters.price_max,
            )
            if len(rows) > 3:
                rows = self._narrow_by_query(rows, filters.raw_query) or rows
        if not rows:
            return None

        lines = [self.format_product(r) for r in rows[:8]]
        head = f"为您找到 {len(rows)} 款商品：" if len(rows) > 1 else "查询结果："
        active = self.active_promotions(
            today, rows[0].get("category") if not skus else None
        )
        if active:
            lines.append(
                "当前有效活动：" + "；".join(
                    f"{p['title']}（{p['discount']}，{p['start_date']}~{p['end_date']}）"
                    for p in active[:3]
                )
            )
        return head + "\n" + "\n".join(lines)
