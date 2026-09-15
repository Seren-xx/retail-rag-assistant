"""RRF 融合：按稳定 doc_id 合并多路召回；concat_fuse 为无融合的消融基线。"""

from .. import config


def rrf_fuse(ranked_lists, k=None):
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank + 1)，k 缺省取 config.rrf_k。"""
    k = k or config.rrf_k
    scores = {}
    for results in ranked_lists:
        for rank, (doc, _s) in enumerate(results):
            if doc.id not in scores:
                scores[doc.id] = [doc, 0.0]
            scores[doc.id][1] += 1.0 / (k + rank + 1)

    ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
    return [(doc, score) for doc, score in ranked]


def concat_fuse(ranked_lists):
    """朴素多路拼接：按名次轮询交替，不做分数融合（消融基线用）。"""
    seen = set()
    merged = []
    iterators = [iter(lst) for lst in ranked_lists]
    while iterators:
        next_round = []
        for it in iterators:
            for doc, _s in it:
                if doc.id not in seen:
                    seen.add(doc.id)
                    merged.append((doc, 0.0))
                    break
            else:
                continue
            next_round.append(it)
        iterators = next_round
    return merged
