"""CrossEncoder 二阶段重排：对 (query, doc) 逐对打分后排序。"""

from .. import config


class CrossEncoderReranker:
    def __init__(self, model_name=None):
        self.model_name = model_name or config.reranker_model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query, hits, top_k=None):
        """hits: [(Document, score)] → 重排后的 [(Document, rerank_score)]，top_k 缺省取 config.rerank_top_k。"""
        if not hits:
            return []
        try:
            model = self._load()
            pairs = [[query, doc.page_content] for doc, _s in hits]
            scores = model.predict(pairs)
            scored = sorted(zip([d for d, _ in hits], scores), key=lambda x: x[1], reverse=True)
            return [(doc, float(score)) for doc, score in scored[: top_k or config.rerank_top_k]]
        except Exception:
            # 模型加载失败时降级：保持原序
            return hits[: top_k or config.rerank_top_k]


def make_reranker():
    """生产重排器：本地 CrossEncoder 模型（config.reranker_model）。"""
    return CrossEncoderReranker()
