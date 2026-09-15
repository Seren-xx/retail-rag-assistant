"""时效处理：过期促销过滤；易变信息按更新时间小幅加成（tie-breaker）。"""

import datetime as dt

from .. import config


def resolve_today(settings=None):
    """评测复现用固定日期（settings.today / config.today）；留空取系统日期。"""
    today = (getattr(settings, "today", None) if settings else None) or config.today
    if today:
        return dt.date.fromisoformat(today)
    return dt.date.today()


def _parse_date(value):
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def filter_expired(hits, today):
    """过滤掉已过期的文档（metadata.end_date < today）。"""
    kept = []
    for doc, score in hits:
        end = _parse_date(doc.metadata.get("end_date", ""))
        if end is not None and end < today:
            continue
        kept.append((doc, score))
    return kept


def apply_freshness_boost(hits, today, weight=None, window_days=None):
    """volatile 文档按 updated_at 乘性加成 score *= (1 + weight * 新鲜度)，负分不加成。"""
    weight = config.freshness_weight if weight is None else weight
    window_days = config.freshness_window_days if window_days is None else window_days
    adjusted = []
    for doc, score in hits:
        if score > 0 and doc.metadata.get("volatile"):
            updated = _parse_date(doc.metadata.get("updated_at", ""))
            if updated is not None:
                days = max(0, (today - updated).days)
                freshness = max(0.0, 1.0 - days / window_days)
                score = score * (1.0 + weight * freshness)
        adjusted.append((doc, score))
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted
