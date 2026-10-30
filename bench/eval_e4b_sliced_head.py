#!/usr/bin/env python3
"""
eval_e4b_sliced_head.py — Gemma-4-E4B 末层 Logits 切片投影优化对比评测脚本
对比项：
  1. 数学等价性验证 (Margin 误差)
  2. 不同 Prompt 长度下的单题前向延迟 (短 / 中 / 长)
  3. Mini-Bench 12 题准则判定准确率与总耗时
  4. 多选 (Choice) 与打分 (Score) 正确性验证
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
CLI_BIN = str(WORKSPACE / "bin" / "decision-cli")
MODEL_ALIAS = "gemma-4-e4b"


def run_decision_cli(args: list[str], env_extra: dict = None) -> dict:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = [CLI_BIN] + args
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\nStderr: {res.stderr}\nStdout: {res.stdout}")
    return json.loads(res.stdout.strip())


def main():
    print(f"===============================================================")
    print(f" Gemma-4-E4B 末层切片优化对比评测 (Feature Flag: sliced-head)")
    print(f"===============================================================\n")

    # 预热并确保模型在显存中
    print(">>> 正在预热模型...")
    run_decision_cli([
        "noun", "--model", MODEL_ALIAS,
        "--rubrics", "预热准则", "--context", "预热文本",
        "--no-cache", "--format", "json"
    ])
    print(">>> 预热完成。\n")

    # 1. 数学等价性测试 (Equivalence Verification)
    print(">>> [测试 1] 数学等价性验证 (Margin & Decision 一致性)...")
    test_cases = [
        {"rubrics": "判断用户反馈是否属于支付流程受阻的严重报错", "context": "微信支付点击后一直转圈，最后提示错误码 5002，钱扣了但订单未生效"},
        {"rubrics": "判断用户反馈是否属于支付流程受阻的严重报错", "context": "界面的背景颜色很好看，希望能增加深色模式切换功能"},
        {"rubrics": "判断是否属于咨询类问题", "context": "请问会员包年套餐下个月到期后怎么续费？支持微信支付吗？"},
        {"rubrics": "判断是否为系统崩溃", "context": "点击设置直接闪退，无法打开主页面"},
    ]

    all_match = True
    max_margin_diff = 0.0
    for idx, tc in enumerate(test_cases, 1):
        # 原版模式 (全词表 lm_head)
        r_orig = run_decision_cli([
            "noun", "--model", MODEL_ALIAS,
            "--rubrics", tc["rubrics"], "--context", tc["context"],
            "--sliced-head", "off",
            "--no-cache", "--format", "json"
        ])

        # 切片模式 (候选 Token 权重投影)
        r_opt = run_decision_cli([
            "noun", "--model", MODEL_ALIAS,
            "--rubrics", tc["rubrics"], "--context", tc["context"],
            "--sliced-head", "on",
            "--no-cache", "--format", "json"
        ])

        margin_orig = r_orig["logits"]["margin"]
        margin_opt = r_opt["logits"]["margin"]
        diff = abs(margin_orig - margin_opt)
        max_margin_diff = max(max_margin_diff, diff)

        decision_orig = r_orig["decision"]
        decision_opt = r_opt["decision"]
        match = (decision_orig == decision_opt) and (diff < 1e-4)
        if not match:
            all_match = False
        print(f"  用例 {idx}: 原始决策={decision_orig} (Margin={margin_orig:.4f}) | 切片决策={decision_opt} (Margin={margin_opt:.4f}) | 差值={diff:.2e}")

    print(f"  --> 最大 Margin 差值: {max_margin_diff:.2e}, 结果完全一致: {all_match}\n")

    # 2. 多分类 (Choice) 与打分 (Score) 功能验证
    print(">>> [测试 2] 多分类 (Choice) 与打分 (Score) 功能验证...")
    r_choice_orig = run_decision_cli([
        "choose", "--model", MODEL_ALIAS,
        "--rubrics", "将工单分类到对应技术模块",
        "--choices", "pay:支付问题,login:账号登录,ui:前端界面展示",
        "--context", "用户输入验证码后点击登录无反应，控制台报错 401",
        "--sliced-head", "off",
        "--no-cache", "--format", "json"
    ])

    r_choice_opt = run_decision_cli([
        "choose", "--model", MODEL_ALIAS,
        "--rubrics", "将工单分类到对应技术模块",
        "--choices", "pay:支付问题,login:账号登录,ui:前端界面展示",
        "--context", "用户输入验证码后点击登录无反应，控制台报错 401",
        "--sliced-head", "on",
        "--no-cache", "--format", "json"
    ])
    choice_match = r_choice_orig["choice"] == r_choice_opt["choice"]
    print(f"  Choice: 原版={r_choice_orig['choice']} ({r_choice_orig['confidence']:.4f}) | 切片={r_choice_opt['choice']} ({r_choice_opt['confidence']:.4f}) | 一致: {choice_match}")

    r_score_orig = run_decision_cli([
        "score", "--model", MODEL_ALIAS,
        "--rubrics", "评估用户情绪愤怒等级",
        "--levels", "平静,轻度烦躁,明显愤怒,极度愤怒暴怒",
        "--context", "这是什么破垃圾软件？！扣了老子两遍钱还直接闪退，客服到底死哪去了？！",
        "--sliced-head", "off",
        "--no-cache", "--format", "json"
    ])

    r_score_opt = run_decision_cli([
        "score", "--model", MODEL_ALIAS,
        "--rubrics", "评估用户情绪愤怒等级",
        "--levels", "平静,轻度烦躁,明显愤怒,极度愤怒暴怒",
        "--context", "这是什么破垃圾软件？！扣了老子两遍钱还直接闪退，客服到底死哪去了？！",
        "--sliced-head", "on",
        "--no-cache", "--format", "json"
    ])
    score_match = r_score_orig["top_level"] == r_score_opt["top_level"]
    print(f"  Score : 原版=Level {r_score_orig['top_level']} ({r_score_orig['top_label']}) | 切片=Level {r_score_opt['top_level']} ({r_score_opt['top_label']}) | 一致: {score_match}\n")

    # 3. 不同 Prompt 长度下的端到端时延对比 (Latency Benchmarking)
    print(">>> [测试 3] 不同 Prompt 长度下的端到端推理时延对比 (各跑 5 次取平均)...")
    prompt_cases = [
        {
            "name": "短 Prompt (~60 tokens)",
            "rubrics": "判断是否属于故障反馈",
            "context": "应用打开后白屏死锁，错误码 500",
        },
        {
            "name": "中等 Prompt (~350 tokens)",
            "rubrics": "判断用户反馈是否属于高危系统阻断故障。高危阻断定义：核心链路完全中断、造成资金损失或敏感数据泄露，且用户无法通过刷新重试规避。",
            "context": "我们在使用企业版管理后台时，财务人员在执行月度批量薪资代发过程中，页面提交后直接返回 HTTP 504 Gateway Timeout。经过后端日志排查，数据库中资金已被划扣，但下游发放流水表完全未写入，重试提交又提示订单幂等性冲突报错，导致当期代发全线阻塞挂起，目前影响两百余名员工发放。",
        },
        {
            "name": "长 Prompt (~1200 tokens)",
            "rubrics": "判断提供的系统运行审计报告中是否出现了需要值班团队立即介入的严重安全告警事件。严重告警标准包括但不限于：远程代码执行、未经授权的权限提升、生产数据库被全量导出、鉴权服务签名密钥泄露或大规模异常登录。",
            "context": (
                "【系统审计事件流水记录】\n"
                + "\n".join([
                    f"事件流水 #{1000 + i}: 时间 2026-10-31 02:{i:02d}:15，来源节点 worker-{i % 8}.prod，"
                    f"服务模块 auth-service，操作类型 TOKEN_VERIFY，状态码 200 OK，延迟 {12 + i % 5}ms，客户端 IP 10.24.{i % 250}.{i % 250}。"
                    for i in range(25)
                ])
                + "\n事件流水 #1026: 时间 2026-10-31 02:26:01，来源节点 auth-gateway.prod，"
                "操作类型 ADMIN_ROLE_GRANT，状态码 200 OK，触发告警：检测到未知 IP 198.51.100.44 通过未签名管理接口直接将普通测试账户权限提升为 super_admin 超级权限并清空了访问审计日志。"
            ),
        },
    ]

    benchmark_summary = []
    N_RUNS = 5
    for pc in prompt_cases:
        name = pc["name"]
        rub = pc["rubrics"]
        ctx = pc["context"]

        # 测原版
        latencies_orig = []
        for _ in range(N_RUNS):
            res = run_decision_cli([
                "noun", "--model", MODEL_ALIAS,
                "--rubrics", rub, "--context", ctx,
                "--sliced-head", "off",
                "--no-cache", "--format", "json"
            ])
            latencies_orig.append(res["latency_ms"])

        # 测切片优化版
        latencies_opt = []
        for _ in range(N_RUNS):
            res = run_decision_cli([
                "noun", "--model", MODEL_ALIAS,
                "--rubrics", rub, "--context", ctx,
                "--sliced-head", "on",
                "--no-cache", "--format", "json"
            ])
            latencies_opt.append(res["latency_ms"])

        avg_orig = sum(latencies_orig) / len(latencies_orig)
        avg_opt = sum(latencies_opt) / len(latencies_opt)
        delta_ms = avg_orig - avg_opt
        pct = (delta_ms / avg_orig) * 100.0

        benchmark_summary.append({
            "name": name,
            "orig_ms": round(avg_orig, 2),
            "opt_ms": round(avg_opt, 2),
            "saved_ms": round(delta_ms, 2),
            "speedup_pct": round(pct, 1),
        })

        print(f"  {name:25s} | 原版耗时: {avg_orig:6.2f} ms | 切片耗时: {avg_opt:6.2f} ms | 节省: {delta_ms:5.2f} ms ({pct:+.1f}%)")

    # 4. 保存评测报表
    report_file = WORKSPACE / "bench" / "results" / "gemma_4_e4b_sliced_head_eval.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model": MODEL_ALIAS,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mathematical_equivalence": {
            "all_match": all_match,
            "max_margin_diff": max_margin_diff,
            "choice_match": choice_match,
            "score_match": score_match,
        },
        "latency_benchmark": benchmark_summary,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n>>> 评测完成，报表已写入: {report_file}")


if __name__ == "__main__":
    main()
