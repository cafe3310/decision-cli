"""Language Benchmark Suite for decision-cli.

Includes:
1. Path A Multi-Class Choice (6 categories, forward & reverse order invariance) and Score (5 ordinal levels).
2. Chinese Binary 12-item Mini Benchmark.
3. Official JevBench 231-item Public Benchmark (Easy, Standard, Hard).
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_DIR.parent
SUBMODULE_DIR = BENCH_DIR / "jevbench"
DATA_DIR = SUBMODULE_DIR / "datasets" / "public"

from bench.internal.dataset_generator import get_path_a_benchmark_data, get_mini_bench_data


def resolve_cli_bin(custom_bin: Optional[str] = None) -> str:
    if custom_bin and Path(custom_bin).exists():
        return str(Path(custom_bin).resolve())
    repo_bin = REPO_ROOT / "bin" / "decision-cli"
    if repo_bin.exists():
        return str(repo_bin)
    found = shutil.which("decision-cli")
    if found:
        return found
    local_bin = Path.home() / ".local" / "bin" / "decision-cli"
    if local_bin.exists():
        return str(local_bin)
    return "decision-cli"


def softmax_dict(logits_map: dict, temp: float = 1.0) -> dict:
    if temp <= 0:
        temp = 1.0
    scaled = {k: float(v) / temp for k, v in logits_map.items()}
    m = max(scaled.values())
    exps = {k: math.exp(v - m) for k, v in scaled.items()}
    s = sum(exps.values())
    return {k: v / s for k, v in exps.items()}


def nrps_single(probs: dict, gold_idx: int, K: int) -> float:
    """Normalized Ranked Probability Score over levels 0..K-1."""
    if K <= 1:
        return 0.0
    cdf_p = 0.0
    rps = 0.0
    for k in range(K - 1):
        cdf_p += float(probs.get(str(k), 0.0))
        cdf_y = 1.0 if gold_idx <= k else 0.0
        rps += (cdf_p - cdf_y) ** 2
    return rps / float(K - 1)


def run_path_a_eval(model: str, cli_bin: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Runs Path A Multi-Class Choice and Score benchmark."""
    data = get_path_a_benchmark_data()
    choice_cases = data["choice_task"]["cases"]
    choice_rubric = data["choice_task"]["rubrics"]
    choices_list = data["choice_task"]["choices"]

    if limit is not None and limit > 0:
        choice_cases = choice_cases[:limit]

    fwd_choices_str = ",".join(f'{c["key"]}:{c["description"]}' for c in choices_list)
    rev_choices_str = ",".join(f'{c["key"]}:{c["description"]}' for c in reversed(choices_list))

    print(f"[{model}] Running Path A Choice Evaluation ({len(choice_cases)} cases x 2 orders)...")
    choice_results = []
    fwd_correct = 0
    rev_correct = 0
    order_consistent = 0

    for case in choice_cases:
        cid = case["id"]
        expected = case["expected"]
        ctx = case["context"]

        cmd_fwd = [
            cli_bin, "choose",
            "--model", model,
            "--rubrics", choice_rubric,
            "--choices", fwd_choices_str,
            "--context", ctx,
            "--format", "json",
        ]
        res_fwd = json.loads(subprocess.check_output(cmd_fwd, text=True))

        cmd_rev = [
            cli_bin, "choose",
            "--model", model,
            "--rubrics", choice_rubric,
            "--choices", rev_choices_str,
            "--context", ctx,
            "--format", "json",
        ]
        res_rev = json.loads(subprocess.check_output(cmd_rev, text=True))

        f_ans = res_fwd["choice"]
        r_ans = res_rev["choice"]

        f_match = (f_ans == expected)
        r_match = (r_ans == expected)
        consistent = (f_ans == r_ans)

        if f_match:
            fwd_correct += 1
        if r_match:
            rev_correct += 1
        if consistent:
            order_consistent += 1

        choice_results.append({
            "id": cid,
            "expected": expected,
            "forward_choice": f_ans,
            "forward_match": f_match,
            "reverse_choice": r_ans,
            "reverse_match": r_match,
            "order_consistent": consistent,
            "forward_top_prob": res_fwd.get("confidence", 0.0),
            "reverse_top_prob": res_rev.get("confidence", 0.0),
        })

    choice_total = len(choice_cases)
    choice_summary = {
        "total": choice_total,
        "forward_acc": round(fwd_correct / choice_total, 4) if choice_total else 0,
        "reverse_acc": round(rev_correct / choice_total, 4) if choice_total else 0,
        "order_consistency_rate": round(order_consistent / choice_total, 4) if choice_total else 0,
        "cases": choice_results,
    }

    # Score Task
    score_cases = data["score_task"]["cases"]
    score_rubric = data["score_task"]["rubrics"]
    score_levels = data["score_task"]["score_levels"]

    if limit is not None and limit > 0:
        score_cases = score_cases[:limit]

    levels_str = ",".join(score_levels)
    print(f"[{model}] Running Path A Score Evaluation ({len(score_cases)} cases)...")
    score_results = []
    score_exact = 0
    total_ae = 0.0

    for case in score_cases:
        cid = case["id"]
        expected_lvl = case["expected_level"]
        ctx = case["context"]

        cmd_score = [
            cli_bin, "score",
            "--model", model,
            "--rubrics", score_rubric,
            "--levels", levels_str,
            "--context", ctx,
            "--format", "json",
        ]
        res = json.loads(subprocess.check_output(cmd_score, text=True))
        pred_lvl = res.get("score", res.get("top_level", 0))
        exact = (pred_lvl == expected_lvl)
        ae = abs(pred_lvl - expected_lvl)

        if exact:
            score_exact += 1
        total_ae += ae

        score_results.append({
            "id": cid,
            "expected_level": expected_lvl,
            "predicted_level": pred_lvl,
            "exact_match": exact,
            "absolute_error": ae,
            "top_prob": res.get("confidence", 0.0),
            "all_level_probs": res.get("probabilities", {}),
        })

    score_total = len(score_cases)
    score_summary = {
        "total": score_total,
        "exact_acc": round(score_exact / score_total, 4) if score_total else 0,
        "mae": round(total_ae / score_total, 4) if score_total else 0,
        "cases": score_results,
    }

    return {
        "model": model,
        "choice_task": choice_summary,
        "score_task": score_summary,
    }


def run_mini_bench_eval(model: str, cli_bin: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Runs 12-item Chinese binary mini-benchmark."""
    cases = get_mini_bench_data()
    if limit is not None and limit > 0:
        cases = cases[:limit]

    print(f"[{model}] Running Chinese Mini-Bench ({len(cases)} items)...")
    results = []
    correct = 0

    for c in cases:
        cmd = [
            cli_bin, "noun",
            "--model", model,
            "--rubrics", c["rubrics"],
            "--context", c["context"],
            "--format", "json",
        ]
        out = json.loads(subprocess.check_output(cmd, text=True))
        pred = out["decision"]
        is_correct = (pred == c["expected"])
        if is_correct:
            correct += 1

        results.append({
            "id": c["id"],
            "expected": c["expected"],
            "predicted": pred,
            "correct": is_correct,
            "confidence": out.get("confidence", 0.0),
            "category": c["category"],
        })

    total = len(cases)
    return {
        "model": model,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "items": results,
    }


def task_to_worker_item(task) -> dict:
    q = task.question
    qtype = q["type"]
    instr = q["instructions"]
    crit = q.get("criteria")

    if isinstance(task.state, str):
        ctx = task.state
    else:
        ctx = json.dumps(task.state, ensure_ascii=False)

    if qtype == "noul":
        rubrics = instr
        if isinstance(crit, dict) and crit:
            parts = []
            for k, lbl in [("true", "YES (true)"), ("yes", "YES (true)"), ("false", "NO (false)"), ("no", "NO (false)")]:
                if k in crit and crit[k]:
                    parts.append(f"- {lbl}: {crit[k]}")
            if parts:
                rubrics = instr.strip() + "\n" + "\n".join(parts)
        return {
            "id": task.id,
            "primitive": "noul",
            "rubrics": rubrics,
            "context": ctx,
        }
    elif qtype == "choice":
        choices = []
        for lbl in task.labels:
            desc = crit.get(lbl, lbl) if isinstance(crit, dict) else lbl
            choices.append({"key": lbl, "description": str(desc)})
        return {
            "id": task.id,
            "primitive": "choice",
            "rubrics": instr,
            "choices": choices,
            "context": ctx,
        }
    elif qtype == "score":
        levels = [str(x) for x in crit] if isinstance(crit, list) else [str(x) for x in task.labels]
        return {
            "id": task.id,
            "primitive": "score",
            "rubrics": instr,
            "score_levels": levels,
            "context": ctx,
        }
    else:
        raise ValueError(f"Unknown qtype: {qtype}")


def normalize_eval_label(val) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s in ("true", "yes", "1", "t", "y"):
        return "true"
    if s in ("false", "no", "0", "f", "n"):
        return "false"
    return s


def run_jevbench_public_eval(model: str, cli_bin: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Runs official JevBench 231-item public evaluation using `decision-cli systemone`."""
    if not (DATA_DIR / "easy.jsonl").exists():
        return {"error": f"JevBench datasets not found at {DATA_DIR}. Please check git submodule."}

    sys.path.insert(0, str(SUBMODULE_DIR))
    from jevbench.tasks import load_jsonl

    tasks_by_tier = {
        "easy": load_jsonl(str(DATA_DIR / "easy.jsonl")),
        "standard": load_jsonl(str(DATA_DIR / "original.jsonl")),
        "hard": load_jsonl(str(DATA_DIR / "hard.jsonl")),
    }

    all_tasks = []
    for tier in ["easy", "standard", "hard"]:
        for t in tasks_by_tier[tier]:
            all_tasks.append((tier, t))

    if limit is not None and limit > 0:
        all_tasks = all_tasks[:limit]

    print(f"[{model}] Running JevBench Public Benchmark ({len(all_tasks)} items via `decision-cli systemone`)...")

    tier_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    confs = []
    corrects = []
    start_t = time.perf_counter()

    for idx, (tier, t) in enumerate(all_tasks):
        q = t.question
        req_obj = {
            "model": model,
            "state": t.state if isinstance(t.state, str) else json.dumps(t.state, ensure_ascii=False),
            "questions": {
                t.id: {
                    "type": q["type"],
                    "instructions": q["instructions"],
                    "criteria": q.get("criteria"),
                }
            }
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f_in:
            json.dump(req_obj, f_in, ensure_ascii=False)
            in_path = f_in.name

        try:
            cmd = [
                cli_bin, "systemone",
                "--input", in_path,
                "--format", "json",
            ]
            res_str = subprocess.check_output(cmd, text=True)
            res = json.loads(res_str)
            ans = res.get("answers", {}).get(t.id, {})

            gold = t.expected
            pred = None
            conf = ans.get("confidence", 0.0)

            if ans.get("type") == "choice":
                pred = ans.get("choice")
            elif ans.get("type") == "noul":
                prob = ans.get("noul", 0.0)
                pred = "true" if prob >= 0.5 else "false"
            elif ans.get("type") == "score":
                pred = str(int(round(ans.get("score", 0.0))))

            is_correct = (normalize_eval_label(pred) == normalize_eval_label(gold)) if gold is not None else False
            tier_stats[tier]["total"] += 1
            if is_correct:
                tier_stats[tier]["correct"] += 1

            confs.append(float(conf))
            corrects.append(1 if is_correct else 0)
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)

    elapsed = time.perf_counter() - start_t
    total_all = len(corrects)
    total_correct = sum(corrects)
    acc = round(total_correct / total_all, 4) if total_all else 0.0

    return {
        "model": model,
        "total_items": total_all,
        "overall_accuracy": acc,
        "elapsed_seconds": round(elapsed, 2),
        "tier_breakdown": {
            tier: {
                "total": st["total"],
                "accuracy": round(st["correct"] / st["total"], 4) if st["total"] else 0,
            }
            for tier, st in tier_stats.items()
        },
    }
