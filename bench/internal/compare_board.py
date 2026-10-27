"""Leaderboard comparison module for decision-cli.

Compares local benchmark results against the official JevBench v1.7.7 live leaderboard.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_DIR.parent
BOARD_177_JSON = BENCH_DIR / "results" / "leaderboard" / "jevbench_v1.7.7_board.json"
DEFAULT_EVAL_JSON = BENCH_DIR / "results" / "jevbench_public_231_eval_results.json"


def compare_with_leaderboard(eval_path: Optional[Path] = None):
    if eval_path is None:
        eval_path = DEFAULT_EVAL_JSON
    if not eval_path.exists():
        print(f"Evaluation file not found: {eval_path}")
        return
    if not BOARD_177_JSON.exists():
        print(f"Leaderboard snapshot not found: {BOARD_177_JSON}")
        return

    rep = json.loads(eval_path.read_text(encoding="utf-8"))
    d177 = json.loads(BOARD_177_JSON.read_text(encoding="utf-8"))
    systems = d177.get("systems", [])

    print("================================================================================")
    print("           JevBench v1.7.7 Official Leaderboard Snapshot Comparison             ")
    print("================================================================================")
    print(f"Total systems on official leaderboard: {len(systems)} (Source SHA256: {d177.get('source_sha256', '')[:12]}...)")

    print("\nTop 5 Systems on Official Board:")
    for s in systems[:5]:
        rank = s.get("rank_composite_A", "?")
        name = s.get("display") or s.get("key") or "unknown"
        score = s.get("jevbench_score") or s.get("capability", "N/A")
        print(f"  #{rank:<2} {name:<42} : {score}")

    print("\nLocal Models Evaluation Summary (on 231 Public JevBench items):")
    for model_name, mdata in rep.get("models", {}).items():
        cfg = mdata.get("default_config", {})
        tier_acc = cfg.get("tier_raw_accuracy", {})
        easy = tier_acc.get("easy", "N/A")
        std = tier_acc.get("standard", "N/A")
        hard = tier_acc.get("hard", "N/A")
        est_score = cfg.get("composite_est_A", "N/A")
        calib = cfg.get("calibration", {})
        ece = calib.get("ece", "N/A")
        p50 = cfg.get("speed", {}).get("p50_ms", "N/A")
        print(f"  * {model_name:<20} | Score Est: {est_score} | Easy: {easy} | Std: {std} | Hard: {hard} | ECE: {ece} | Latency p50: {p50}ms")
    print("================================================================================")


if __name__ == "__main__":
    compare_with_leaderboard()
