"""问答服务：数值/事实查询走 catalog 确定性回答，语义查询走混合检索 + 门禁（未通过拒答）。"""

import os
import re
import time
from dataclasses import dataclass, field

from .. import config
from ..config import load_settings
from ..indexing.index_service import IndexService
from ..knowledge.catalog import Catalog
from ..query.constraints import (
    content_tokens,
    expand_query,
    extract_filters,
    merge_filters,
    sanitize_llm_filters,
)
from ..query.router import IntentRouter
from ..retrieval.bm25 import BM25Index
from ..retrieval.context import build_context
from ..retrieval.dense import make_dense_index, make_embedder
from ..retrieval.freshness import apply_freshness_boost, filter_expired, resolve_today
from ..retrieval.fusion import rrf_fuse
from ..retrieval.quality import evaluate_quality
from ..retrieval.rerank import CrossEncoderReranker, make_reranker
from .llm import make_filter_extractor, make_llm
from .session import make_session_store

REFUSAL_MESSAGES = {
    "no_candidate": "抱歉，知识库中没有找到与您问题相关的资料，请您换个说法，或联系人工客服。",
    "low_retrieval_confidence": "抱歉，目前知识库中的资料不足以回答这个问题，建议您换个说法或联系人工客服。",
    "low_rerank_confidence": "抱歉，目前知识库中的资料不足以回答这个问题，建议您换个说法或联系人工客服。",
    "time_mismatch": "您询问的活动时间与当前有效促销不一致，请确认活动时间或查看最新活动。",
}

# 促销意图：答案在促销文档里；CrossEncoder 会把品类匹配的商品文档排前
# （如"乳制品优惠"→ 酸奶商品），需按意图把促销文档提到商品之前。
PROMO_INTENT = re.compile(r"活动|优惠|促销|会员日|满减|打折|立减|满\d+减\d+")


@dataclass
class RetrieveResult:
    """单次检索的完整输出（决策 + 质量 + 证据）。"""

    query: str
    expanded_query: str
    route: str
    filters: dict
    decision: str
    quality: dict
    evidence: list = field(default_factory=list)
    context: str = ""
    elapsed_ms: float = 0.0


class QAService:
    def __init__(self, settings=None, embedder=None, reranker=None, llm=None, session_store=None, force_sync=False):
        self.s = settings or load_settings()
        self.catalog = Catalog.load(self.s)

        # 索引：首次运行自动同步，之后增量
        self.index_service = IndexService(self.s)
        if force_sync or not os.path.exists(self.index_service.docstore_path):
            self.index_service.sync()
        self.docs = self.index_service.load_docs()

        # 检索栈
        self.embedder = embedder or make_embedder()
        self.dense = make_dense_index(self.embedder, self.s.index_dir).load_or_build(
            self.s.index_dir, self.docs
        )
        self.bm25 = BM25Index()
        self.bm25.build(self.docs)
        self.reranker = reranker or make_reranker()
        self.has_rerank = isinstance(self.reranker, CrossEncoderReranker)

        self.router = IntentRouter()
        self.sessions = session_store or make_session_store(self.s.redis_url)
        self.llm = llm or make_llm(self.s)

        # 查询约束：规则提取为底座，LLM 补充（未配置 Key 时为 None，纯规则）
        self.filter_extractor = make_filter_extractor(self.s)
        self._extract_failures = 0
        self._extract_disabled = False

    # ---------- 查询理解 ----------

    def _extract_filters(self, query):
        """规则 + 词典提取为底座；LLM 结果经白名单清洗后合并，LLM 不得推翻规则。

        LLM 连续失败 2 次自动熔断回退纯规则，保证延迟稳定、链路不中断。
        """
        filters = extract_filters(query)
        if self.filter_extractor is None or self._extract_disabled:
            return filters
        raw = self.filter_extractor(query)
        if raw is None:
            self._extract_failures += 1
            if self._extract_failures >= 2:
                self._extract_disabled = True
            return filters
        self._extract_failures = 0
        return merge_filters(filters, sanitize_llm_filters(raw))

    # ---------- 检索 ----------

    def _prefilter(self, hits, filters):
        """品牌/品类/价格硬过滤：仅作用于结构化商品文档，政策文档不受影响。

        多品牌比较（如"蒙牛和伊利"）取并集过滤，不会错杀成单一品牌。
        """
        if not (filters.brands or filters.category or filters.price_min is not None or filters.price_max is not None):
            return hits
        kept = []
        for doc, score in hits:
            if doc.metadata.get("kind") == "product":
                if filters.brands and doc.metadata.get("brand") not in filters.brands:
                    continue
                if filters.category and doc.metadata.get("category") != filters.category:
                    continue
                price = doc.metadata.get("price")
                if isinstance(price, (int, float)):
                    if filters.price_min is not None and price < filters.price_min:
                        continue
                    if filters.price_max is not None and price > filters.price_max:
                        continue
            kept.append((doc, score))
        return kept

    @staticmethod
    def _promote_filtered(hits, filters):
        """带品类/价格约束时，符合条件商品文档排在无关文档之前。"""
        if not (filters.category or filters.price_min is not None or filters.price_max is not None):
            return hits
        products, others = [], []
        for doc, score in hits:
            (products if doc.metadata.get("kind") == "product" else others).append((doc, score))
        return products + others

    @staticmethod
    def _promote_promotions(hits, query, filters):
        """促销意图 + 品类/时间约束明确的查询：促销文档提到商品之前。

        泛化查询（如"生日有什么优惠"，答案可能在会员规则）不提升，由词面补救兜底。
        filter_expired 已在前一步执行，此处剩余促销文档均为当前生效。
        """
        if not PROMO_INTENT.search(query):
            return hits
        if not (filters.category or filters.months):
            return hits
        promos, others = [], []
        for doc, score in hits:
            (promos if doc.metadata.get("kind") == "promotion" else others).append((doc, score))
        return promos + others

    def retrieve(self, query, top_k=None):
        """混合检索 → 重排 → 时效 → 门禁 → 上下文组装（检索参数读 config 常量）。"""
        t0 = time.perf_counter()
        today = resolve_today(self.s)

        route_type, _num = self.router.route(query)
        filters = self._extract_filters(query)
        expanded = expand_query(query)

        bm25_hits = self._prefilter(self.bm25.search(expanded), filters)
        dense_hits = self._prefilter(self.dense.search(expanded), filters)

        fused = rrf_fuse([bm25_hits, dense_hits])
        reranked = self.reranker.rerank(query, fused[:config.rerank_top_k])

        # 时效：过期促销直接剔除；volatile 文档小幅新鲜度加成
        reranked = filter_expired(reranked, today)
        reranked = apply_freshness_boost(reranked, today)
        reranked = self._promote_filtered(reranked, filters)
        reranked = self._promote_promotions(reranked, query, filters)

        quality = evaluate_quality(
            query, bm25_hits, dense_hits, reranked, today,
            has_rerank=self.has_rerank,
            promo_intent=bool(PROMO_INTENT.search(query)),
            promo_category=filters.category,
        )

        context = ""
        if quality.ok:
            context, _ = build_context(reranked)

        evidence = [
            {
                "doc_id": d.id,
                "source": d.metadata.get("source", ""),
                "score": round(float(score), 4),
            }
            for d, score in reranked[: top_k or config.final_top_k]
        ]

        return RetrieveResult(
            query=query,
            expanded_query=expanded,
            route=route_type,
            filters={
                "brands": filters.brands,
                "category": filters.category,
                "price_min": filters.price_min,
                "price_max": filters.price_max,
                "months": filters.months,
            },
            decision=quality.decision,
            quality={
                "agreement": round(quality.agreement, 4),
                "coverage": round(quality.coverage, 4),
                "word_score": round(quality.word_score, 4),
                "rerank_conf": None if quality.rerank_conf is None else round(quality.rerank_conf, 4),
                "reasons": quality.reasons,
            },
            evidence=evidence,
            context=context,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ---------- 问答 ----------

    def _product_evidence(self, filters, skus=None):
        """数值通路的证据：来自结构化目录，不经检索。"""
        rows = self.catalog.filter_products(
            brands=filters.brands,
            category=filters.category,
            price_min=filters.price_min,
            price_max=filters.price_max,
        )
        if skus:
            rows = [r for r in (self.catalog.get(s) for s in skus) if r]
        return [
            {"doc_id": f"product:{r.get('product_id')}", "source": "products.csv", "score": None}
            for r in rows[:8]
        ]

    def answer(self, query, session_id=None):
        """问答入口：数值查询走目录直答，语义查询走检索门禁 + LLM/证据直出。"""
        t0 = time.perf_counter()
        route_type, num_constraints = self.router.route(query)
        today = resolve_today(self.s)

        # 通路一：数值/事实查询走结构化目录，数字确定性生成
        if route_type == "numeric":
            filters = self._extract_filters(query)
            text = self.catalog.answer_numeric(
                filters, today, skus=num_constraints.get("skus")
            )
            if text:
                out = {
                    "answer": text,
                    "route": "numeric",
                    "decision": "ok",
                    "quality": None,
                    "evidence": self._product_evidence(filters, num_constraints.get("skus")),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
                self._remember(session_id, query, out)
                return out

        # 通路二：语义检索 + 质量门禁
        res = self.retrieve(query)
        if res.decision != "ok":
            answer = REFUSAL_MESSAGES.get(res.decision, REFUSAL_MESSAGES["low_retrieval_confidence"])
        elif self.llm is not None:
            history = self.sessions.get(session_id) if session_id else []
            answer = self.llm(query=query, context=res.context, history=history)
        else:
            # 未配置 LLM：直接返回通过门禁的证据，保证链路可独立运行
            answer = "根据知识库资料：\n\n" + res.context

        out = {
            "answer": answer,
            "route": res.route,
            "decision": res.decision,
            "quality": res.quality,
            "evidence": res.evidence,
            "elapsed_ms": res.elapsed_ms,
        }
        self._remember(session_id, query, out)
        return out

    def _remember(self, session_id, query, result):
        """会话写入（Redis），未提供 session_id 时跳过。"""
        if not session_id:
            return
        self.sessions.add(session_id, "user", query)
        self.sessions.add(session_id, "assistant", str(result.get("answer", "")))
