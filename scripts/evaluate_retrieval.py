"""运行检索消融评测：python scripts/evaluate_retrieval.py --today 2026-09-11。

使用生产默认模型（BGE-M3 / bge-reranker-v2-m3 / ChromaDB），
与线上问答完全同构，指标即生产行为。
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from retail_assistant.config import load_settings  # noqa: E402
from retail_assistant.evaluation.ablation import (
    build_qa,
    evaluate,
    format_summary,
    save_reports,
)  # noqa: E402
from retail_assistant.evaluation.dataset import load_cases  # noqa: E402
from retail_assistant.indexing.index_service import IndexService  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="检索消融评测")
    parser.add_argument("--modes", default="dense,bm25,concat,rrf,full")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--llm-filters", action="store_true",
                        help="开启 Qwen3-Max 约束抽取对比（默认关闭，保证离线可复现）")
    parser.add_argument("--cases", default=None, help="标注查询集 jsonl 路径")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--today", default=None, help="固定评测日期，如 2026-09-11")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true", help="评测前强制重建索引")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    settings = load_settings()
    if args.embedding_model:
        settings.embedding_model = args.embedding_model
    if args.reranker_model:
        settings.reranker_model = args.reranker_model
    if args.data_dir:
        settings.data_dir = args.data_dir
    if args.index_dir:
        settings.index_dir = args.index_dir
    if args.today:
        settings.today = args.today
    # 评测口径：默认关闭 LLM 约束提取（--llm-filters 可开启对比）
    settings.llm_filters = bool(args.llm_filters)

    if args.rebuild:
        stats = IndexService(settings).sync()
        print("index rebuilt:", {k: v for k, v in stats.items() if k != "files"})

    qa = build_qa(settings)
    cases_path = args.cases or os.path.join(_ROOT, "eval", "retrieval_cases.jsonl")
    cases = load_cases(cases_path)
    print(f"eval cases: {len(cases)} | active docs: {len(qa.docs)}")

    summary, records = evaluate(qa, cases, args.modes.split(","), top_k=args.top_k)
    print(format_summary(summary))

    csv_path, json_path = save_reports(summary, records, args.output_dir)
    print(f"\nreports saved:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
