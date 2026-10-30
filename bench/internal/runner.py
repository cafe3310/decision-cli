"""Unified benchmark runner entrypoint for decision-cli.

Routes suites, models, limit parameters, executes tests via CLI/Worker,
and outputs/saves summary metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

from bench.internal.eval_language import (
    resolve_cli_bin,
    run_path_a_eval,
    run_mini_bench_eval,
    run_jevbench_public_eval,
)
from bench.internal.eval_multimodal import run_multimodal_eval
from bench.internal.compare_board import compare_with_leaderboard

DEFAULT_MODELS = ["gemma-4-26b-a4b", "gemma-4-e4b", "qwen3-reranker-4b", "ling-3.0-tiny"]
MULTIMODAL_MODELS = ["gemma-4-26b-a4b", "gemma-4-e4b"]


def main():
    parser = argparse.ArgumentParser(
        prog="bench/run_bench",
        description="decision-cli Comprehensive Local Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./bench/run_bench --suite language --model gemma-4-26b-a4b
  ./bench/run_bench --suite language --limit 2
  ./bench/run_bench --suite multimodal --model gemma-4-26b-a4b --limit 5
  ./bench/run_bench --suite all --model all
  ./bench/run_bench --compare-board
""",
    )
    parser.add_argument(
        "-s", "--suite",
        choices=["language", "multimodal", "all"],
        default="all",
        help="Evaluation suite: 'language' (text/choice/score/jevbench), 'multimodal' (image/doc/ui), or 'all' (default: all)",
    )
    parser.add_argument(
        "-m", "--model",
        default="gemma-4-26b-a4b",
        help="Target model alias: 'gemma-4-26b-a4b', 'gemma-4-e4b', 'qwen3-reranker-4b', 'all', or comma-separated list (default: gemma-4-26b-a4b)",
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation cases per task (useful for smoke tests and rapid iteration)",
    )
    parser.add_argument(
        "--compare-board",
        action="store_true",
        help="Display comparison against the official JevBench v1.7.7 live leaderboard snapshot",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="bench/results",
        help="Directory to save evaluation raw results (default: bench/results)",
    )
    parser.add_argument(
        "--cli-bin",
        default=None,
        help="Path to decision-cli binary (default: automatically detected from ./bin/decision-cli or PATH)",
    )

    args = parser.parse_args()

    if args.compare_board and args.suite == "all" and args.model == "gemma-4-26b-a4b" and len(sys.argv) == 2:
        compare_with_leaderboard()
        return

    cli_bin = resolve_cli_bin(args.cli_bin)
    print(f"================================================================================")
    print(f"                   decision-cli Local Benchmark Suite                          ")
    print(f"================================================================================")
    print(f"CLI Binary : {cli_bin}")
    print(f"Suite      : {args.suite}")
    print(f"Model(s)   : {args.model}")
    print(f"Limit      : {args.limit if args.limit else 'None (Full Run)'}")
    print(f"Output Dir : {args.output_dir}")
    print(f"================================================================================\n")

    if args.model.lower() == "all":
        target_models = DEFAULT_MODELS
    else:
        target_models = [m.strip() for m in args.model.split(",") if m.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = {}

    for model in target_models:
        print(f"\n>>>>>>>> Target Model: {model} <<<<<<<<")
        model_results = {}

        if args.suite in ("language", "all"):
            # 1. Mini-bench
            mini_res = run_mini_bench_eval(model, cli_bin, limit=args.limit)
            model_results["mini_bench"] = mini_res
            print(f"  -> Mini-bench Accuracy: {mini_res.get('accuracy', 0.0) * 100:.1f}% ({mini_res.get('correct')}/{mini_res.get('total')})")

            # 2. Path A (Choice & Score)
            path_a_res = run_path_a_eval(model, cli_bin, limit=args.limit)
            model_results["path_a"] = path_a_res
            c_task = path_a_res.get("choice_task", {})
            s_task = path_a_res.get("score_task", {})
            print(f"  -> Path A Choice Fwd Acc: {c_task.get('forward_acc', 0.0) * 100:.1f}%, Rev Acc: {c_task.get('reverse_acc', 0.0) * 100:.1f}%, Order Invariance: {c_task.get('order_consistency_rate', 0.0) * 100:.1f}%")
            print(f"  -> Path A Score Exact Acc: {s_task.get('exact_acc', 0.0) * 100:.1f}%, MAE: {s_task.get('mae', 0.0)}")

            # 3. JevBench Public 231 (if submodule present)
            jev_res = run_jevbench_public_eval(model, cli_bin, limit=args.limit)
            if "error" not in jev_res:
                model_results["jevbench_public"] = jev_res
                print(f"  -> JevBench Public Accuracy: {jev_res.get('overall_accuracy', 0.0) * 100:.1f}% ({jev_res.get('total_items')} items)")
            else:
                print(f"  -> [Notice] JevBench Public skipped: {jev_res['error']}")

        if args.suite in ("multimodal", "all"):
            if model in MULTIMODAL_MODELS:
                mm_res = run_multimodal_eval(model, cli_bin, limit=args.limit)
                if "error" not in mm_res:
                    model_results["multimodal"] = mm_res
                    tracks = mm_res.get("tracks", {})
                    for tname, tdata in tracks.items():
                        print(f"  -> Multimodal {tname} Acc: {tdata.get('accuracy', 0.0) * 100:.1f}% ({tdata.get('correct')}/{tdata.get('total')})")
                else:
                    print(f"  -> [Notice] Multimodal skipped: {mm_res['error']}")
            else:
                print(f"  -> [Skip] Multimodal suite skipped for text-only model: {model}")

        summary_records[model] = model_results

    # Save summary json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"bench_run_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, ensure_ascii=False, indent=2)

    print(f"\n================================================================================")
    print(f"Benchmark run complete! Results written to: {out_file}")
    print(f"================================================================================")

    if args.compare_board:
        compare_with_leaderboard()


if __name__ == "__main__":
    main()
