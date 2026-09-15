"""消融评测：同一查询集对比 Dense / BM25 / Concat / RRF / Full 五种模式，度量每个组件的收益。"""

import csv
import json
import os
import time

from ..application.qa_service import QAService
from ..config import load_settings
from ..evaluation.dataset import EvalCase
from ..evaluation.metrics import hit_at_k, ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from ..retrieval.freshness import resolve_today
from ..retrieval.fusion import concat_fuse, rrf_fuse
from ..query.constraints import expand_query, extract_filters

MODES = ["dense", "bm25", "concat", "rrf", "full"]


def build_qa(settings=None) -> QAService:
    """构建评测用 QAService（生产模型链路，与线上完全同构）。"""
    return QAService(settings=settings or load_settings())


def _rank_mode(qa: QAService, mode: str, expanded: str, filters, top_k: int):
    """非 full 模式的排序：不经过重排与门禁（消融变量隔离）。"""
    s = qa.s
    bm25_hits = qa._prefilter(qa.bm25.search(expanded, s.bm25_top_k), filters)
    dense_hits = qa._prefilter(qa.dense.search(expanded, s.dense_top_k), filters)
    if mode == "bm25":
        fused = bm25_hits
    elif mode == "dense":
        fused = dense_hits
    elif mode == "concat":
        fused = concat_fuse([bm25_hits, dense_hits])
    else:  # rrf
        fused = rrf_fuse([bm25_hits, dense_hits], s.rrf_k)
    return fused[:top_k]


def evaluate(qa: QAService, cases: list[EvalCase], modes: list[str] | None = None, top_k: int = 5):
    """返回 (summary, records)。summary 为模式级指标，records 为逐条明细。"""
    modes = modes or MODES
    answerable = [c for c in cases if not c.is_no_answer]
    no_answer = [c for c in cases if c.is_no_answer]

    summary = {}
    records = []

    for mode in modes:
        latencies = []
        recalls, hits, mrrs, ndcgs = [], [], [], []
        refused_answerable = 0

        for case in answerable:
            filters = extract_filters(case.query)
            expanded = expand_query(case.query)
            t0 = time.perf_counter()
            if mode == "full":
                # 与生产 answer() 一致：数值查询走结构化目录，不经检索门禁
                route_type, num = qa.router.route(case.query)
                numeric_evidence = []
                if route_type == "numeric":
                    numeric_evidence = qa._product_evidence(filters, num.get("skus"))
                if numeric_evidence:
                    ranked_ids = [e["doc_id"] for e in numeric_evidence]
                    decision, quality = "ok", None
                else:
                    res = qa.retrieve(case.query, top_k=top_k)
                    ranked_ids = [e["doc_id"] for e in res.evidence]
                    decision, quality = res.decision, res.quality
            else:
                fused = _rank_mode(qa, mode, expanded, filters, top_k)
                ranked_ids = [d.id for d, _ in fused]
                decision, quality = None, None
            latencies.append((time.perf_counter() - t0) * 1000)

            rr = reciprocal_rank(ranked_ids, case.relevant_ids)
            recalls.append(recall_at_k(ranked_ids, case.relevant_ids, top_k))
            hits.append(hit_at_k(ranked_ids, case.relevant_ids, top_k))
            mrrs.append(rr)
            ndcgs.append(ndcg_at_k(ranked_ids, case.relevant_ids, top_k))
            if decision is not None and decision != "ok":
                refused_answerable += 1

            records.append(
                {
                    "mode": mode,
                    "query": case.query,
                    "tag": case.tag,
                    "relevant_ids": case.relevant_ids,
                    "ranked_ids": ranked_ids,
                    "reciprocal_rank": rr,
                    "decision": decision,
                    "quality": quality,
                    "latency_ms": round(latencies[-1], 2),
                }
            )

        # 无答案用例：只有 full 模式有门禁决策；记录 quality 便于校准阈值
        no_answer_refused = 0
        for case in no_answer:
            decision, quality = None, None
            if mode == "full":
                res = qa.retrieve(case.query, top_k=top_k)
                decision = res.decision
                quality = res.quality
                latencies.append(res.elapsed_ms)
            if decision is not None and decision != "ok":
                no_answer_refused += 1
            records.append(
                {
                    "mode": mode,
                    "query": case.query,
                    "tag": case.tag,
                    "relevant_ids": [],
                    "ranked_ids": [],
                    "reciprocal_rank": 0.0,
                    "decision": decision,
                    "quality": quality,
                    "latency_ms": 0.0,
                }
            )

        summary[mode] = {
            "queries_answerable": len(answerable),
            "queries_no_answer": len(no_answer),
            f"recall_at_{top_k}": round(sum(recalls) / max(len(recalls), 1), 4),
            f"hit_at_{top_k}": round(sum(hits) / max(len(hits), 1), 4),
            "mrr": round(sum(mrrs) / max(len(mrrs), 1), 4),
            f"ndcg_at_{top_k}": round(sum(ndcgs) / max(len(ndcgs), 1), 4),
            "p50_ms": round(percentile(latencies, 0.5), 2),
            "p95_ms": round(percentile(latencies, 0.95), 2),
            "no_answer_accuracy": round(no_answer_refused / max(len(no_answer), 1), 4)
            if mode == "full"
            else None,
            "false_rejection_rate": round(refused_answerable / max(len(answerable), 1), 4)
            if mode == "full"
            else None,
        }

    return summary, records


def save_reports(summary: dict, records: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "retrieval_ablation.csv")
    fields = [
        "mode", "queries_answerable", "queries_no_answer",
        "recall_at_5", "hit_at_5", "mrr", "ndcg_at_5",
        "p50_ms", "p95_ms", "no_answer_accuracy", "false_rejection_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for mode, metrics in summary.items():
            writer.writerow({"mode": mode, **metrics})

    json_path = os.path.join(out_dir, "retrieval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def format_summary(summary: dict) -> str:
    """打印用：对齐的消融表。"""
    header = (
        f"{'mode':<8} {'recall@5':>9} {'hit@5':>7} {'MRR':>7} {'nDCG@5':>8} "
        f"{'P50(ms)':>9} {'P95(ms)':>10} {'no-ans':>7} {'f-rej':>7}"
    )
    lines = [header, "-" * len(header)]
    for mode, m in summary.items():
        lines.append(
            f"{mode:<8} {m['recall_at_5']:>9.4f} {m['hit_at_5']:>7.4f} {m['mrr']:>7.4f} "
            f"{m['ndcg_at_5']:>8.4f} {m['p50_ms']:>9.2f} {m['p95_ms']:>10.2f} "
            f"{str(m['no_answer_accuracy']):>7} {str(m['false_rejection_rate']):>7}"
        )
    return "\n".join(lines)
