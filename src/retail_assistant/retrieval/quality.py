"""检索质量门禁：词面信号 + 重排置信 + 时效匹配 → 拒答/放行决策。"""

import re
from dataclasses import dataclass, field
from datetime import date

from .. import config
from ..query.constraints import content_tokens


@dataclass
class QualityReport:
    """门禁决策与信号分数。"""

    decision: str = ""
    reasons: list = field(default_factory=list)
    agreement: float = 0.0
    coverage: float = 0.0
    word_score: float = 0.0
    rerank_conf: float | None = None

    @property
    def ok(self):
        return self.decision == "ok"


def evaluate_quality(
    query,
    bm25_hits,
    dense_hits,
    ranked_hits,
    today=None,
    has_rerank=False,
    promo_intent=False,
    promo_category=None,
):
    """三个信号依次判定：词面过低 → 拒；重排置信不足 → 拒；月份不符 → 拒。阈值取 config。"""
    if not ranked_hits:
        return QualityReport(decision="no_candidate", reasons=["候选为空"])

    top_doc = ranked_hits[0][0]
    top_text = top_doc.page_content.lower()

    # 信号 1：词面 = 0.5*双路前5重合度 + 0.5*内容词覆盖率（纯功能词查询置 1.0）
    bm25_ids = {d.id for d, _ in bm25_hits[:5]}
    dense_ids = {d.id for d, _ in dense_hits[:5]}
    denom = max(1, min(5, len(bm25_ids), len(dense_ids)))
    agreement = len(bm25_ids & dense_ids) / denom

    tokens = content_tokens(query)
    if tokens:
        coverage = sum(1 for t in tokens if t in top_text) / len(tokens)
    else:
        coverage = 1.0
    word_score = 0.5 * agreement + 0.5 * coverage

    report = QualityReport(agreement=agreement, coverage=coverage, word_score=word_score)

    if word_score < config.word_score_floor:
        # 补救：词面低但候选前 3 重排置信高（同义改写字面不匹配，语义相关）
        if has_rerank and ranked_hits:
            top3_max = max(float(score) for _, score in ranked_hits[:3])
            if top3_max >= config.rerank_rescue:
                report.rerank_conf = top3_max
                report.decision = "ok"
                report.reasons.append(
                    f"词面信号 {word_score:.2f} 偏低，候选前3最高重排置信 {top3_max:.2f} 补救"
                )
                return report
        report.decision = "low_retrieval_confidence"
        report.reasons.append(f"词面信号 {word_score:.2f} 低于阈值 {config.word_score_floor}")
        return report

    # 信号 2：重排置信（top-1 过低且覆盖不足 → 拒，带三类豁免/补救）
    if has_rerank and ranked_hits:
        report.rerank_conf = float(ranked_hits[0][1])
        # 促销意图豁免：CrossEncoder 对短促销文本打分系统性偏低，
        # 要求候选中存在查询品类的促销文档（全品类也可），防"汽油优惠"搭便车。
        has_promo_match = bool(promo_category) and any(
            d.metadata.get("kind") == "promotion"
            and d.metadata.get("category") in (promo_category, "全品类")
            for d, _ in ranked_hits
        )
        # 品类浏览豁免：查询带品类约束且候选前 3 存在该品类商品文档
        # （如"门店有哪些饮料"→饮料商品列表）。仅非促销意图查询适用：
        # "零食满99减15还有吗"式促销提问，商品高分不构成回答依据。
        has_category_product = bool(promo_category) and any(
            d.metadata.get("kind") == "product"
            and d.metadata.get("category") == promo_category
            for d, _ in ranked_hits
        )
        if report.rerank_conf < config.rerank_floor and not (
            (promo_intent and has_promo_match)
            or (not promo_intent and has_category_product)
        ):
            # 补救分数只看知识/促销文档（正确文档可能被重排挤到第 2、3 位）
            knowledge_max = max(
                (float(score) for d, score in ranked_hits[:3]
                 if d.metadata.get("kind") in ("promotion", "knowledge")),
                default=0.0,
            )
            # 覆盖完整的短口语查询豁免（如"怎么开发票"每个字都命中）
            if coverage < config.rerank_coverage_floor and coverage < 1.0:
                if knowledge_max >= config.rerank_rescue:
                    report.rerank_conf = knowledge_max
                    report.reasons.append(
                        f"top-1 重排置信不足，候选前3知识/促销文档最高 {knowledge_max:.2f} 补救"
                    )
                else:
                    report.decision = "low_rerank_confidence"
                    report.reasons.append(
                        f"重排置信 {report.rerank_conf:.3f} 低于阈值 {config.rerank_floor} 且覆盖不足"
                    )
                    return report

    # 信号 3：查询问"N月"活动时，top-1 促销必须在该月生效
    months = [int(m) for m in re.findall(r"(\d{1,2})月", query)]
    if months:
        start = _parse_iso(top_doc.metadata.get("start_date"))
        end = _parse_iso(top_doc.metadata.get("end_date"))
        if start is not None and end is not None:
            ref_year = (today or date.today()).year
            month_ok = any(start <= date(ref_year, m, 1) <= end for m in months)
            if not month_ok:
                report.decision = "time_mismatch"
                report.reasons.append(
                    f"查询月份 {months} 与 top-1 促销生效期 {start}~{end} 不符"
                )
                return report

    report.decision = "ok"
    return report


def _parse_iso(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None
