"""检索评测指标：Recall@K、Hit@K、MRR、nDCG@K 与延迟分位数（只评检索，不评生成文风）。"""

import math


def recall_at_k(ranked_ids: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & set(relevant)) / len(set(relevant))


def hit_at_k(ranked_ids: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(ranked_ids[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: list[str]) -> float:
    relevant = set(relevant)
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: list[str], k: int = 5) -> float:
    """二元相关性的 nDCG@K。"""
    if not relevant:
        return 0.0
    relevant = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(ranked_ids[:k])
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数；空列表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
