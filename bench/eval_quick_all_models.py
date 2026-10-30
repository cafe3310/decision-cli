#!/usr/bin/env python3
"""
eval_quick_all_models.py — 快速测试 4 个模型在纯文本及多模态下的切片优化耗时与效果差异
"""

import json
import os
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
CLI_BIN = str(WORKSPACE / "bin" / "decision-cli")
TEST_IMG = str(WORKSPACE / "tests" / "fixtures" / "red_alert.png")

TEXT_MODELS = [
    ("ling-3.0-tiny", "判断是否为系统故障反馈", "数据库主从同步中断报错 1045"),
    ("qwen3-reranker-4b", "判断是否为系统故障反馈", "数据库主从同步中断报错 1045"),
    ("gemma-4-e4b", "判断是否为系统故障反馈", "数据库主从同步中断报错 1045"),
    ("gemma-4-26b-a4b", "判断是否为系统故障反馈", "数据库主从同步中断报错 1045"),
]

MM_MODELS = [
    ("gemma-4-e4b", "判断图片中是否包含红色警告标志", "监控大盘状态截图如下", TEST_IMG),
    ("gemma-4-26b-a4b", "判断图片中是否包含红色警告标志", "监控大盘状态截图如下", TEST_IMG),
]


def run_cli(args: list[str], env_extra: dict = None) -> dict:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = [CLI_BIN] + args
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\nStderr: {res.stderr}\nStdout: {res.stdout}")
    return json.loads(res.stdout.strip())


def main():
    print("===============================================================")
    print(" 四模型纯文本 & 多模态切片优化对比快速测试")
    print("===============================================================\n")

    # 1. 四模型纯文本测试
    print(">>> 1. 四模型纯文本二元判定测试 (短文本，各跑 3 次均值)...")
    text_results = []
    for model_name, rub, ctx in TEXT_MODELS:
        print(f"  正在测试 {model_name}...")
        # 对称预热 (off 与 on 各预热 1 次，避免 Metal 算子首次 JIT 编译影响稳态测速)
        _ = run_cli(["noun", "--model", model_name, "--rubrics", rub, "--context", ctx, "--sliced-head", "off", "--no-cache", "--format", "json"])
        _ = run_cli(["noun", "--model", model_name, "--rubrics", rub, "--context", ctx, "--sliced-head", "on", "--no-cache", "--format", "json"])

        # 原版 (3次)
        l_orig = []
        d_orig = None
        m_orig = 0.0
        for _ in range(3):
            r = run_cli([
                "noun", "--model", model_name, "--rubrics", rub, "--context", ctx,
                "--sliced-head", "off", "--no-cache", "--format", "json"
            ])
            l_orig.append(r["latency_ms"])
            d_orig = r["decision"]
            m_orig = r["logits"]["margin"]

        # 切片版 (3次)
        l_opt = []
        d_opt = None
        m_opt = 0.0
        for _ in range(3):
            r = run_cli([
                "noun", "--model", model_name, "--rubrics", rub, "--context", ctx,
                "--sliced-head", "on", "--no-cache", "--format", "json"
            ])
            l_opt.append(r["latency_ms"])
            d_opt = r["decision"]
            m_opt = r["logits"]["margin"]

        avg_orig = sum(l_orig) / len(l_orig)
        avg_opt = sum(l_opt) / len(l_opt)
        diff_m = abs(m_orig - m_opt)
        delta_lat = avg_orig - avg_opt
        pct = (delta_lat / avg_orig) * 100.0 if avg_orig > 0 else 0.0

        item = {
            "model": model_name,
            "decision_orig": d_orig,
            "decision_opt": d_opt,
            "margin_diff": diff_m,
            "lat_orig_ms": round(avg_orig, 2),
            "lat_opt_ms": round(avg_opt, 2),
            "lat_saved_ms": round(delta_lat, 2),
            "speedup_pct": round(pct, 1),
        }
        text_results.append(item)
        print(f"    原版: {avg_orig:6.2f} ms | 切片: {avg_opt:6.2f} ms | 节省: {delta_lat:5.2f} ms ({pct:+.1f}%) | 决策一致: {d_orig == d_opt} (ΔMargin={diff_m:.2e})")

    # 2. 多模态图文输入测试
    print("\n>>> 2. 多模态 (Vision+Text) 图文判定测试 (各跑 3 次均值)...")
    mm_results = []
    for model_name, rub, ctx, img in MM_MODELS:
        print(f"  正在测试 {model_name} 多模态...")
        # 对称预热 1 次
        _ = run_cli(["noun", "--model", model_name, "--rubrics", rub, "--context", ctx, "--image", img, "--sliced-head", "off", "--no-cache", "--format", "json"])
        _ = run_cli(["noun", "--model", model_name, "--rubrics", rub, "--context", ctx, "--image", img, "--sliced-head", "on", "--no-cache", "--format", "json"])

        l_orig = []
        d_orig = None
        m_orig = 0.0
        for _ in range(3):
            r = run_cli([
                "noun", "--model", model_name, "--rubrics", rub, "--context", ctx,
                "--image", img, "--sliced-head", "off", "--no-cache", "--format", "json"
            ])
            l_orig.append(r["latency_ms"])
            d_orig = r["decision"]
            m_orig = r["logits"]["margin"]

        l_opt = []
        d_opt = None
        m_opt = 0.0
        for _ in range(3):
            r = run_cli([
                "noun", "--model", model_name, "--rubrics", rub, "--context", ctx,
                "--image", img, "--sliced-head", "on", "--no-cache", "--format", "json"
            ])
            l_opt.append(r["latency_ms"])
            d_opt = r["decision"]
            m_opt = r["logits"]["margin"]

        avg_orig = sum(l_orig) / len(l_orig)
        avg_opt = sum(l_opt) / len(l_opt)
        diff_m = abs(m_orig - m_opt)
        delta_lat = avg_orig - avg_opt
        pct = (delta_lat / avg_orig) * 100.0 if avg_orig > 0 else 0.0

        item = {
            "model": model_name,
            "decision_orig": d_orig,
            "decision_opt": d_opt,
            "margin_diff": diff_m,
            "lat_orig_ms": round(avg_orig, 2),
            "lat_opt_ms": round(avg_opt, 2),
            "lat_saved_ms": round(delta_lat, 2),
            "speedup_pct": round(pct, 1),
        }
        mm_results.append(item)
        print(f"    原版: {avg_orig:6.2f} ms | 切片: {avg_opt:6.2f} ms | 节省: {delta_lat:5.2f} ms ({pct:+.1f}%) | 决策一致: {d_orig == d_opt} (ΔMargin={diff_m:.2e})")

    out_file = WORKSPACE / "bench" / "results" / "quick_all_models_sliced_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"text": text_results, "multimodal": mm_results}, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 快速测试完成，结果已保存到: {out_file}")


if __name__ == "__main__":
    main()
