"""Multimodal Benchmark Suite for decision-cli.

Evaluates multimodal models (gemma-4-26b-a4b, gemma-4-e4b) on ImageJev-box:
- Track 1 (noul): Visual Target & UI Verification (binary yes/no)
- Track 2 (choice): 4-Way Category / UI Classification
- Track 3 (choice): 4-Way Spatial Grounding / Bounding-Box Selection
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_DIR.parent
SUBMODULE_DIR = BENCH_DIR / "jevbench"
BOX_DIR = SUBMODULE_DIR / "datasets" / "imagejev-box-v0.1"

DISJOINT_COCO_POOL = [
    ("giraffe", "the giraffe"),
    ("zebra", "the zebra"),
    ("elephant", "the elephant"),
    ("bear", "the bear"),
    ("airplane", "the airplane"),
    ("train", "the train"),
    ("fire hydrant", "the fire hydrant"),
    ("toilet", "the toilet"),
    ("pizza", "the pizza"),
    ("broccoli", "the broccoli"),
    ("sheep", "the sheep"),
    ("horse", "the horse"),
    ("cat", "the cat"),
    ("dog", "the dog"),
]


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


def item_descriptor(item: dict) -> str:
    fam = item["family"]
    target = item["instructions"].replace(" in `photo`", "").strip()
    prov = item.get("provenance", {})
    if fam == "coco":
        cat = prov.get("category", "object")
        return f"a photo showing {target} (category: {cat})"
    elif fam == "doc":
        tpl = prov.get("template", "document")
        return f"a {tpl} document containing {target}"
    elif fam == "ui":
        tpl = prov.get("template", "UI")
        return f"a {tpl} interface screen containing {target}"
    return target


def pick_disjoint_distractors(item: dict, all_items_by_fam: dict, k: int, seed: int) -> list:
    rng = random.Random(seed)
    fam = item["family"]
    prov = item.get("provenance", {})
    if fam == "coco":
        cur_cat = prov.get("category", "")
        candidates = [
            f"a photo showing {desc} (category: {cat})"
            for cat, desc in DISJOINT_COCO_POOL
            if cat != cur_cat
        ]
        rng.shuffle(candidates)
        return candidates[:k]
    else:
        cur_tpl = prov.get("template", "")
        by_tpl = {}
        for other in all_items_by_fam[fam]:
            otpl = other.get("provenance", {}).get("template", "")
            if otpl and otpl != cur_tpl and otpl not in by_tpl:
                by_tpl[otpl] = item_descriptor(other)
        candidates = list(by_tpl.values())
        rng.shuffle(candidates)
        return candidates[:k]


def fmt_box(b: dict) -> str:
    ymin = int(round(b["top"] * 1000))
    xmin = int(round(b["left"] * 1000))
    ymax = int(round(b["bottom"] * 1000))
    xmax = int(round(b["right"] * 1000))
    cx, cy = (b["left"] + b["right"]) / 2.0, (b["top"] + b["bottom"]) / 2.0
    v_pos = "top" if cy < 0.38 else ("bottom" if cy > 0.62 else "middle")
    h_pos = "left" if cx < 0.38 else ("right" if cx > 0.62 else "center")
    return f"[{ymin}, {xmin}, {ymax}, {xmax}] ({v_pos}-{h_pos})"


def make_distractor_boxes(g: dict) -> list:
    w = min(0.38, max(0.08, g["right"] - g["left"]))
    h = min(0.38, max(0.08, g["bottom"] - g["top"]))
    cx, cy = (g["left"] + g["right"]) / 2.0, (g["top"] + g["bottom"]) / 2.0
    anchors = [
        (0.18, 0.18),
        (0.82, 0.18),
        (0.18, 0.82),
        (0.82, 0.82),
        (0.50, 0.15),
        (0.50, 0.85),
    ]
    anchors.sort(key=lambda a: (a[0] - cx) ** 2 + (a[1] - cy) ** 2, reverse=True)
    out = []
    for ax, ay in anchors[:3]:
        l = max(0.0, min(1.0 - w, ax - w / 2.0))
        t = max(0.0, min(1.0 - h, ay - h / 2.0))
        out.append({"left": l, "top": t, "right": l + w, "bottom": t + h})
    return out


def build_multimodal_benchmark(items: list) -> Dict[str, list]:
    by_fam = defaultdict(list)
    for it in items:
        by_fam[it["family"]].append(it)

    track_noul = []
    track_classify = []
    track_grounding = []
    labels_4 = ["A", "B", "C", "D"]

    for idx, it in enumerate(items):
        img_abs = str(BOX_DIR / it["image"])
        fam = it["family"]
        is_pos = (idx % 2 == 0)

        # Track 1
        if is_pos:
            rub = f"Verify whether this image shows {item_descriptor(it)}."
            exp = "true"
        else:
            dist = pick_disjoint_distractors(it, by_fam, 1, seed=idx * 17 + 3)[0]
            rub = f"Verify whether this image shows {dist}."
            exp = "false"

        track_noul.append({
            "id": f"mm_noul_{idx:03d}_{it['id']}",
            "primitive": "noul",
            "rubrics": rub,
            "context": f"Image: {it['image']} ({fam})",
            "image": img_abs,
            "expected": exp,
            "family": fam,
        })

        # Track 2
        gold_desc = item_descriptor(it)
        dist_3 = pick_disjoint_distractors(it, by_fam, 3, seed=idx * 31 + 7)
        while len(dist_3) < 3:
            dist_3.append(f"an unrelated placeholder {len(dist_3)}")
        c_opts = [gold_desc] + dist_3
        rng_c = random.Random(idx * 43 + 11)
        rng_c.shuffle(c_opts)
        g_idx_c = c_opts.index(gold_desc)
        gold_key_c = labels_4[g_idx_c]

        track_classify.append({
            "id": f"mm_cls_{idx:03d}_{it['id']}",
            "primitive": "choice",
            "rubrics": "Which description best matches this image?",
            "choices": [{"key": labels_4[i], "description": c_opts[i]} for i in range(4)],
            "context": f"Image: {it['image']}",
            "image": img_abs,
            "expected": gold_key_c,
            "family": fam,
        })

        # Track 3
        gold_box_fmt = fmt_box(it["gold"])
        d_boxes = make_distractor_boxes(it["gold"])
        b_opts = [gold_box_fmt] + [fmt_box(b) for b in d_boxes]
        rng_b = random.Random(idx * 79 + 13)
        rng_b.shuffle(b_opts)
        g_idx_b = b_opts.index(gold_box_fmt)
        gold_key_b = labels_4[g_idx_b]

        target_desc = it["instructions"].replace(" in `photo`", "").strip()
        track_grounding.append({
            "id": f"mm_box_{idx:03d}_{it['id']}",
            "primitive": "choice",
            "rubrics": f"Which bounding box [ymin, xmin, ymax, xmax] tightly covers '{target_desc}'?",
            "choices": [{"key": labels_4[i], "description": b_opts[i]} for i in range(4)],
            "context": f"Image: {it['image']}",
            "image": img_abs,
            "expected": gold_key_b,
            "family": fam,
        })

    return {
        "track_1_noul": track_noul,
        "track_2_classify": track_classify,
        "track_3_grounding": track_grounding,
    }


def run_multimodal_eval(model: str, cli_bin: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Runs multimodal benchmark on ImageJev-box."""
    meta_path = BOX_DIR / "items.jsonl"
    if not meta_path.exists():
        return {"error": f"ImageJev-box dataset not found at {meta_path}. Please check git submodule."}

    with open(meta_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if limit is not None and limit > 0:
        items = items[:limit]

    print(f"[{model}] Running Multimodal Benchmark on {len(items)} images (3 tracks)...")
    tracks = build_multimodal_benchmark(items)
    track_results = {}

    for track_name, track_items in tracks.items():
        print(f"[{model}] Evaluating {track_name} ({len(track_items)} items)...")
        correct = 0
        start_t = time.perf_counter()

        for item in track_items:
            prim = item["primitive"]
            img_path = item["image"]
            rubric = item["rubrics"]
            ctx = item["context"]
            gold = item["expected"]

            if prim == "noul":
                cmd = [
                    cli_bin, "noun",
                    "--model", model,
                    "--rubrics", rubric,
                    "--image", img_path,
                    "--context", ctx,
                    "--format", "json",
                ]
                res_str = subprocess.check_output(cmd, text=True)
                res = json.loads(res_str)
                # "是" represents true, "否" represents false
                ans_str = "true" if res.get("decision") == "是" else "false"
                if ans_str == str(gold).lower():
                    correct += 1
            elif prim == "choice":
                # Choices format: key:description with Chinese comma for any nested commas
                choices_parts = []
                for c in item["choices"]:
                    desc = c["description"].replace(",", "，")
                    choices_parts.append(f"{c['key']}:{desc}")
                choices_str = ",".join(choices_parts)

                cmd = [
                    cli_bin, "choose",
                    "--model", model,
                    "--rubrics", rubric,
                    "--choices", choices_str,
                    "--image", img_path,
                    "--context", ctx,
                    "--format", "json",
                ]
                res_str = subprocess.check_output(cmd, text=True)
                res = json.loads(res_str)
                choice_ans = res.get("choice", "")
                if choice_ans.upper() == str(gold).upper():
                    correct += 1

        elapsed = time.perf_counter() - start_t
        total = len(track_items)
        acc = round(correct / total, 4) if total else 0.0
        track_results[track_name] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "elapsed_seconds": round(elapsed, 2),
        }

    return {
        "model": model,
        "tracks": track_results,
    }
