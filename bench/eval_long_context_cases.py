#!/usr/bin/env python3
"""
eval_long_context_cases.py — 合成长上下文用例并在不同模型上评测切片优化效果
长上下文梯度：
  1. 场景 A: ~1,500 tokens (分布式集群服务日志审计流水)
  2. 场景 B: ~3,500 tokens (企业数据合规安全审计报告与策略比对)
  3. 场景 C: ~6,000 tokens (微服务全链路故障排查事件与复盘报告)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
CLI_BIN = str(WORKSPACE / "bin" / "decision-cli")


def generate_long_contexts():
    cases = []

    # -------------------------------------------------------------
    # 场景 A: 约 1,500 tokens (分布式集群系统运行与异常审计)
    # -------------------------------------------------------------
    rubric_a = (
        "判断系统日志流中是否出现了需要值班团队立即介入的 P0 级严重故障事件。"
        "P0 故障标准严格定义为：核心存储引擎主节点掉线导致脑裂、数据分区丢失不可恢复、"
        "或鉴权中心私钥在日志中明文打印暴露。"
    )
    lines_a = [
        f"[{time.strftime('%Y-%m-%d')} 01:{i % 60:02d}:{i % 59:02d}.{100 + i % 800}] "
        f"node=k8s-worker-{i % 16:02d}.prod.internal cluster=shanghai-az{1 + i % 3} "
        f"component=etcd-raft-cluster module=raft-core v=3.5.12 "
        f"event=HEARTBEAT_ACK peer_id={1001 + (i * 7) % 19} term=12984 latency_us={230 + (i * 13) % 400} "
        f"status=HEALTHY sync_index={9823410 + i * 42} commit_index={9823400 + i * 42} "
        f"msg='Raft follower keepalive confirmed within configured election timeout boundary'"
        for i in range(40)
    ]
    # 注入关键异常
    lines_a.append(
        f"[{time.strftime('%Y-%m-%d')} 01:52:19.492] "
        f"node=k8s-worker-03.prod.internal cluster=shanghai-az1 "
        f"component=etcd-raft-cluster module=raft-election "
        f"event=LEADER_NETWORK_PARTITION_CONFIRMED status=FATAL "
        f"msg='CRITICAL: Storage quorum lost. Raft split-brain detected across shanghai-az1 and az2. "
        f"Primary database engine partition tables lost and unrecoverable, immediate failover halted!'"
    )
    for i in range(15):
        lines_a.append(
            f"[{time.strftime('%Y-%m-%d')} 01:53:{i:02d}.102] "
            f"node=k8s-worker-08.prod.internal cluster=shanghai-az2 "
            f"component=storage-wal-sync module=wal-writer "
            f"event=FLUSH_FAILED status=ERROR err_code=E_DISK_SYNC_IO_FAIL "
            f"msg='WAL segment flush rejected due to missing active storage leader'"
        )
    ctx_a = "\n".join(lines_a)
    cases.append({"id": "case_a_1500_tokens", "name": "场景 A (~1,500 tokens 存储脑裂审计)", "rubrics": rubric_a, "context": ctx_a})

    # -------------------------------------------------------------
    # 场景 B: 约 3,500 tokens (企业数据安全合规长文本比对)
    # -------------------------------------------------------------
    rubric_b = (
        "判断待审员工操作审计记录中是否违反了《金融级数据出境与高密合规安全规范》。"
        "违规标准：在未经二级审批的情况下，将包含员工或客户个人敏感信息（如明文身份证、银行卡、薪酬明细）"
        "的数据包通过未加密通道或个人云盘上传到跨境外部网络。"
    )
    paragraphs_b = [
        "【企业数据安全分级保护规范说明（修订版）】\n"
        "第一章：总则与安全域划分。依据金融监管要求，企业内部数据划分为 L1 公开级、L2 内部受限级、L3 核心机密级以及 L4 最高涉密级。"
        "凡涉及员工薪酬清单、财务账户流水明细、个人实名认证身份原件信息的数据资产，统一定义为 L3 级以上高密资产。"
        "任何人员严禁在未经信息安全部门与合规总监双重书面签批的前提下将此类资产转移出受信生产内网。"
    ]
    for i in range(1, 26):
        paragraphs_b.append(
            f"【日常合规操作流水记录 #{2000 + i}】\n"
            f"操作员工：emp_00{i % 40 + 10}（岗位：业务运维组初级工程师）\n"
            f"接入终端：VDI-SH-SEC-{i:03d}（MAC: 02:42:AC:11:00:{i % 90 + 10:02X}，IP: 10.200.4.{i % 200 + 1}）\n"
            f"操作行为描述：通过受信 Bastion 跳板机访问内部知识库与发布系统，执行常规应用日常巡检工作。"
            f"巡检日志输出约 {i * 128} 行，网络数据包校验均在受控企业防火墙内白名单内，"
            f"未发生任何外联端口或未知域名访问事件，合规检测系统评级：NORMAL (Risk Score: 0.0{i % 8})。"
        )
    # 注入违规事实
    paragraphs_b.append(
        "【日常合规操作流水记录 #2099 - 关键异常条目】\n"
        "操作员工：emp_0088（岗位：财务系统外包开发支持）\n"
        "接入终端：LAPTOP-FIN-DEV（接入网络：访客外部 Wi-Fi，IP: 192.168.10.88）\n"
        "操作行为描述：该员工通过个人账号导出包含 12,450 条明文客户身份证号、招商银行储蓄卡号及本季度全员实名薪资奖金明细的压缩包 (salary_q3_all_plaintext.zip.enc 未加密)，"
        "在未经安全部门审批的情况下，直接通过浏览器将文件上传至海外个人云盘 https://drive.personal-cloud.external/upload/。"
    )
    ctx_b = "\n\n".join(paragraphs_b)
    cases.append({"id": "case_b_3500_tokens", "name": "场景 B (~3,500 tokens 数据跨境违规合规)", "rubrics": rubric_b, "context": ctx_b})

    # -------------------------------------------------------------
    # 场景 C: 约 6,000 tokens (微服务全链路复杂故障排查全流程 Postmortem)
    # -------------------------------------------------------------
    rubric_c = (
        "判断系统故障复盘报告中，故障的根本触发诱因（Root Cause）是否属于【上游第三方支付通道系统级宕机】。"
        "若是由于自身内部代码死锁、配置错误、数据库容量不足或自身变更发布缺陷导致的，则判断为“否”；"
        "仅当故障是由于外部合作银行/第三方通道完全不可用直接引发时才判定为“是”。"
    )
    sections_c = [
        "【故障复盘与事故分析终版报告 (P0 Incident Postmortem)】\n"
        "事故编号：INC-20261030-PAY-OUTAGE\n"
        "影响等级：P0 核心业务中断\n"
        "故障起止时间：2026-10-30 20:15:00 至 2026-10-30 22:45:00 (持续 150 分钟)\n"
        "受影响服务：交易网关 (Trade-GW)、收银台 (Cashier-App)、代收代付中心 (Settle-Hub)\n"
        "故障现象简述：用户在手机客户端点击‘立即支付’后，收银台长时间处于等待状态，随后提示‘系统繁忙，请稍后再试’。"
    ]
    for sec_idx in range(1, 40):
        sections_c.append(
            f"【故障时间线与排查节点记录 Phase {sec_idx}】\n"
            f"时间戳：2026-10-30 20:{sec_idx:02d}:00\n"
            f"排查团队：第 {sec_idx % 4 + 1} 应急响应小组（基础设施 SRE 与分布式存储专项组）\n"
            f"排查动作：排查 Pod 资源负载、JVM GC 暂停耗时、TCP 连接池存活状态以及微服务间 RPC 通信追踪。\n"
            f"指标详情：Pod CPU 利用率稳定在 {20 + sec_idx % 25}%，内存占用 {40 + sec_idx % 15}%，无 OOMKilled 事件发生。"
            f"网络接口卡 eth0 吞吐流速 {120 + sec_idx * 5} MB/s，丢包率 0.000%，交换机背板无拥塞流控帧。"
            f"应用框架连接池当前活跃连接数 {250 + sec_idx * 10}，等待队列长度 {10 + sec_idx % 8}，内部网络通讯正常无异常丢包。"
        )
    # 注入真实根因（自身内部代码缺陷引起连接池耗尽，而非第三方外部通道宕机）
    sections_c.append(
        "【事故根本原因根因分析 (Root Cause Analysis - RCA)】\n"
        "在当晚 20:10:00，研发团队发布了收银台核心组件 v2.14.0 更新，新增了风控灰度打点逻辑。\n"
        "该版本在调用内部风控 RPC 时，错误地在长事务数据库连接内部开启了未设置超时时间的同步等待，导致所有的 HikariCP 数据库连接池被同步锁死。\n"
        "上游银联与网联等外部第三方支付通道在全过程网络探测与交易模拟中始终保持 100% 正常连通（外部通道可用性 100.0%），"
        "本次故障完全由自身收银台代码中的事务级连接池死锁引起，非外部支付通道原因。"
    )
    for sec_idx in range(10):
        sections_c.append(
            f"【后续改进与整改预防措施 Item {sec_idx + 1}】\n"
            f"责任团队：交易中台组。整改内容：全面梳理所有在数据库长事务内的同步 RPC 外部调用，强制添加 200ms 熔断保护与降级回退机制。"
        )
    ctx_c = "\n\n".join(sections_c)
    cases.append({"id": "case_c_6000_tokens", "name": "场景 C (~6,000 tokens 故障复盘根因归因)", "rubrics": rubric_c, "context": ctx_c})

    return cases


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
    print("==================================================================")
    print(" 长上下文用例合成与切片优化性能评测 (1.5k / 3.5k / 6k Tokens)")
    print("==================================================================\n")

    cases = generate_long_contexts()
    for c in cases:
        print(f"  已合成用例: {c['name']} (文本字符数: {len(c['context'])} 字符)")
    print()

    # 测试的模型列表：选取主力 26B MoE、轻量 4.5B Dense、以及混合线性注意力的 Ling 3.0
    models_to_test = [
        "gemma-4-e4b",
        "gemma-4-26b-a4b",
        "ling-3.0-tiny",
    ]

    all_results = []

    for model_name in models_to_test:
        print(f"\n==============================================================")
        print(f" >>> 正在评测模型: {model_name}")
        print(f"==============================================================")

        # 切换模型后先进行对称预热，排除首次加载与算子编译开销
        _ = run_cli(["noun", "--model", model_name, "--rubrics", "预热准则", "--context", "预热文本", "--sliced-head", "off", "--no-cache", "--format", "json"])
        _ = run_cli(["noun", "--model", model_name, "--rubrics", "预热准则", "--context", "预热文本", "--sliced-head", "on", "--no-cache", "--format", "json"])

        for case in cases:
            c_name = case["name"]
            rub = case["rubrics"]
            ctx = case["context"]

            print(f"\n  --- 正在运行: {c_name} ---")

            # 1. 原版 (未开启切片)
            t0 = time.perf_counter()
            r_orig = run_cli([
                "noun", "--model", model_name,
                "--rubrics", rub, "--context", ctx,
                "--sliced-head", "off",
                "--no-cache", "--format", "json"
            ])
            wall_orig_ms = (time.perf_counter() - t0) * 1000.0

            # 2. 切片优化版
            t0 = time.perf_counter()
            r_opt = run_cli([
                "noun", "--model", model_name,
                "--rubrics", rub, "--context", ctx,
                "--sliced-head", "on",
                "--no-cache", "--format", "json"
            ])
            wall_opt_ms = (time.perf_counter() - t0) * 1000.0

            lat_orig = r_orig["latency_ms"]
            lat_opt = r_opt["latency_ms"]
            saved_ms = lat_orig - lat_opt
            pct = (saved_ms / lat_orig) * 100.0 if lat_orig > 0 else 0.0

            dec_orig = r_orig["decision"]
            dec_opt = r_opt["decision"]
            margin_orig = r_orig["logits"]["margin"]
            margin_opt = r_opt["logits"]["margin"]
            diff_m = abs(margin_orig - margin_opt)
            dec_match = dec_orig == dec_opt

            res_record = {
                "model": model_name,
                "case": c_name,
                "context_len_chars": len(ctx),
                "decision_orig": dec_orig,
                "decision_opt": dec_opt,
                "decision_match": dec_match,
                "margin_orig": margin_orig,
                "margin_opt": margin_opt,
                "margin_diff": diff_m,
                "lat_orig_ms": lat_orig,
                "lat_opt_ms": lat_opt,
                "saved_ms": round(saved_ms, 2),
                "speedup_pct": round(pct, 1),
            }
            all_results.append(res_record)

            print(f"    决策结果: 原版={dec_orig} (Margin={margin_orig:.3f}) | 切片={dec_opt} (Margin={margin_opt:.3f}) | 一致={dec_match}")
            print(f"    推理耗时: 原版={lat_orig:7.2f} ms | 切片={lat_opt:7.2f} ms | 净省={saved_ms:7.2f} ms ({pct:+.1f}%)")

    out_file = WORKSPACE / "bench" / "results" / "long_context_cases_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n==============================================================")
    print(f" 全部测试完成！评测报告已保存至: {out_file}")
    print(f"==============================================================")


if __name__ == "__main__":
    main()
