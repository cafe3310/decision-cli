#!/usr/bin/env python3
"""
run_full_eval.py — 全量本地模型与 Jev 综合评测与硬件资源占用采集器

评测范围：
1. 本地四模型（100% 真实前向推理，跳过缓存）：
   - gemma-4-26b-a4b (26B MoE, 3.8B 激活, 支持文本 + 多模态)
   - gemma-4-e4b     (4.5B Dense, 支持文本 + 多模态)
   - ling-3.0-tiny   (7.9B MoE, 1.3B 激活, 纯文本)
   - qwen3-reranker-4b (4.0B Dense, 纯文本)
2. 远端对比基线：
   - TypeSafe Jev 1.13.0 & Imajev-4B (官方榜单基准参考)

采集指标（6 个维度）：
1. 文本分数：JevBench Public 231 题 (Overall / Easy / Standard / Hard + Noul / Choice / Score)、中文 Mini-Bench 12 题、Path A 6分类 (正序/逆序/一致性) & 5档量规打分 (Exact / MAE)
2. 多模态分数（仅对支持的模型）：ImageJev-box 视觉决策集（Track 1 存在性核验 / Track 2 4选1分类 / Track 3 候选框空间定位）
3. 推理速度（文本）：单题平均耗时 (ms)、p50 (ms)、p95 (ms)、吞吐量 (decisions/s)
4. 推理速度（多模态）：冷启动特征编码+前向耗时 (ms)、视觉缓存命中前向耗时 (ms)、综合平均耗时与吞吐量
5. 评测期间进程内存占用（平均、峰值，仅本地）：实时采样 mlx_worker.py + decision-cli bar + decision-cli CLI 的真实物理内存 (phys_footprint)
6. 评测期间 GPU 占用（平均、峰值，仅本地）：实时采样 IOAccelerator 的 Device Utilization % 与 Renderer Utilization %
"""

import ctypes
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "bench" / "jevbench"))

from bench.internal.dataset_generator import get_mini_bench_data, get_path_a_benchmark_data
from bench.internal.eval_multimodal import build_multimodal_benchmark
from jevbench.tasks import load_jsonl

CLI_BIN = str(WORKSPACE / "bin" / "decision-cli")
SOCK_PATH = os.path.expanduser("~/.cache/decision-cli/worker.sock")
STATUS_PATH = os.path.expanduser("~/.cache/decision-cli/worker_status.json")
DATA_DIR = WORKSPACE / "bench" / "jevbench" / "datasets" / "public"
BOX_DIR = WORKSPACE / "bench" / "jevbench" / "datasets" / "imagejev-box-v0.1"

LOCAL_MODELS = [
    {
        "alias": "gemma-4-26b-a4b",
        "arch": "26B MoE (3.8B 激活, 4-bit MLX)",
        "prompt_style": "gemma4-binary",
        "supports_vision": True,
        "pos_tokens": ["是", "Yes", "yes", "YES"],
        "neg_tokens": ["否", "No", "no", "NO"],
        "t_noul": 4.5,
        "t_choice": 3.5,
        "t_score": 1.0,
    },
    {
        "alias": "gemma-4-e4b",
        "arch": "4.5B Dense (4-bit MLX)",
        "prompt_style": "gemma4-binary",
        "supports_vision": True,
        "pos_tokens": ["是", "Yes", "yes", "YES"],
        "neg_tokens": ["否", "No", "no", "NO"],
        "t_noul": 3.5,
        "t_choice": 3.5,
        "t_score": 1.5,
    },
    {
        "alias": "ling-3.0-tiny",
        "arch": "7.9B MoE (1.3B 激活, KDA+MLA 4-bit)",
        "prompt_style": "ling-binary",
        "supports_vision": False,
        "pos_tokens": ["是", "Yes", "yes", "YES"],
        "neg_tokens": ["否", "No", "no", "NO"],
        "t_noul": 1.5,
        "t_choice": 2.0,
        "t_score": 1.5,
    },
    {
        "alias": "qwen3-reranker-4b",
        "arch": "4.0B Dense (Cross-Encoder 4-bit)",
        "prompt_style": "qwen3-reranker",
        "supports_vision": False,
        "pos_tokens": ["yes", "Yes", "YES", "是"],
        "neg_tokens": ["no", "No", "NO", "否"],
        "t_noul": 1.0,
        "t_choice": 3.5,
        "t_score": 2.5,
    },
]

# =========================================================================
# macOS 原生 libproc 物理内存 (phys_footprint) 与 IOAccelerator GPU 采样器
# =========================================================================

libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
RUSAGE_INFO_V4 = 4


class rusage_info_v4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


def get_pid_footprint_mb(pid: int) -> float:
    if pid <= 0:
        return 0.0
    ri = rusage_info_v4()
    res = libproc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(ri))
    if res == 0:
        return ri.ri_phys_footprint / (1024.0 * 1024.0)
    return 0.0


def sample_gpu_stats() -> dict:
    try:
        out = subprocess.check_output(
            ["/usr/sbin/ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if '"PerformanceStatistics"' in line:
                dev_m = re.search(r'"Device Utilization %"=(\d+)', line)
                ren_m = re.search(r'"Renderer Utilization %"=(\d+)', line)
                til_m = re.search(r'"Tiler Utilization %"=(\d+)', line)
                mem_m = re.search(r'"In use system memory"=(\d+)', line)
                dev = float(dev_m.group(1)) if dev_m else 0.0
                ren = float(ren_m.group(1)) if ren_m else 0.0
                til = float(til_m.group(1)) if til_m else 0.0
                mem_mb = float(mem_m.group(1)) / (1024.0 * 1024.0) if mem_m else 0.0
                return {
                    "device_util": max(dev, ren),
                    "renderer_util": ren,
                    "tiler_util": til,
                    "gpu_mem_mb": mem_mb,
                }
    except Exception:
        pass
    return {"device_util": 0.0, "renderer_util": 0.0, "tiler_util": 0.0, "gpu_mem_mb": 0.0}


def find_decision_pids() -> dict:
    """Returns {'worker': [pids], 'bar': [pids], 'cli': [pids]}."""
    pids = {"worker": [], "bar": [], "cli": []}
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                st = json.load(f)
                wpid = int(st.get("pid", 0))
                if wpid > 0:
                    pids["worker"].append(wpid)
    except Exception:
        pass

    try:
        out = subprocess.check_output(["/bin/ps", "-ax", "-o", "pid=,command="], text=True)
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            cmd = parts[1]
            if "mlx_worker.py" in cmd:
                if pid not in pids["worker"]:
                    pids["worker"].append(pid)
            elif "decision-cli bar" in cmd or cmd.endswith("/decision-cli bar"):
                pids["bar"].append(pid)
            elif "decision-cli" in cmd and "run_full_eval" not in cmd:
                pids["cli"].append(pid)
    except Exception:
        pass
    return pids


class HardwareMonitor:
    def __init__(self, interval_sec: float = 0.12):
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self.mem_samples = []  # dicts of worker_mb, bar_mb, cli_mb, total_mb
        self.gpu_samples = []  # dicts of device_util, renderer_util, gpu_mem_mb

    def start(self):
        self._stop.clear()
        self.mem_samples.clear()
        self.gpu_samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            pids = find_decision_pids()
            w_mb = sum(get_pid_footprint_mb(p) for p in pids["worker"])
            b_mb = sum(get_pid_footprint_mb(p) for p in pids["bar"])
            c_mb = sum(get_pid_footprint_mb(p) for p in pids["cli"])
            tot_mb = w_mb + b_mb + c_mb
            if tot_mb > 0:
                self.mem_samples.append({
                    "worker_mb": w_mb,
                    "bar_mb": b_mb,
                    "cli_mb": c_mb,
                    "total_mb": tot_mb,
                })
            g = sample_gpu_stats()
            self.gpu_samples.append(g)
            time.sleep(self.interval_sec)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else 0.0

        def peak(lst):
            return round(max(lst), 1) if lst else 0.0

        w_list = [x["worker_mb"] for x in self.mem_samples if x["worker_mb"] > 0]
        b_list = [x["bar_mb"] for x in self.mem_samples if x["bar_mb"] > 0]
        c_list = [x["cli_mb"] for x in self.mem_samples]
        t_list = [x["total_mb"] for x in self.mem_samples if x["total_mb"] > 0]

        # Filter active GPU samples (> 5% util) if available so brief startup/teardown doesn't dilute active inference GPU load
        dev_all = [x["device_util"] for x in self.gpu_samples]
        dev_active = [x for x in dev_all if x >= 5.0] or dev_all
        ren_all = [x["renderer_util"] for x in self.gpu_samples]
        ren_active = [x for x in ren_all if x >= 5.0] or ren_all
        gmem_all = [x["gpu_mem_mb"] for x in self.gpu_samples]

        return {
            "samples_count": len(self.gpu_samples),
            "memory": {
                "worker_avg_mb": avg(w_list),
                "worker_peak_mb": peak(w_list),
                "bar_avg_mb": avg(b_list),
                "bar_peak_mb": peak(b_list),
                "cli_avg_mb": avg(c_list),
                "cli_peak_mb": peak(c_list),
                "total_avg_mb": avg(t_list),
                "total_peak_mb": peak(t_list),
            },
            "gpu": {
                "device_avg_pct": avg(dev_active),
                "device_peak_pct": peak(dev_all),
                "renderer_avg_pct": avg(ren_active),
                "renderer_peak_pct": peak(ren_all),
                "driver_mem_avg_mb": avg(gmem_all),
                "driver_mem_peak_mb": peak(gmem_all),
            },
        }


# =========================================================================
# 辅助函数：通过 decision-cli 与常驻 Socket 执行无缓存批量推理
# =========================================================================

def resolve_local_model_path(alias: str) -> str:
    out = subprocess.check_output([CLI_BIN, "models", "--format", "json"], text=True)
    data = json.loads(out)
    models = data.get("models", []) if isinstance(data, dict) else data
    for m in models:
        if m["alias"] == alias:
            return m["local_path"]
    raise RuntimeError(f"Model {alias} not found in decision-cli models")


def send_worker_batch(req_payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(600.0)
        s.connect(SOCK_PATH)
        s.sendall((json.dumps(req_payload, ensure_ascii=False) + "\n").encode("utf-8"))
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            if b"\n" in chunk:
                break
        return json.loads(buf.decode("utf-8").strip())


def softmax_dict(logits_map: dict, temp: float = 1.0) -> dict:
    if temp <= 0:
        temp = 1.0
    scaled = {k: float(v) / temp for k, v in logits_map.items()}
    m = max(scaled.values())
    exps = {k: math.exp(v - m) for k, v in scaled.items()}
    s = sum(exps.values())
    return {k: v / s for k, v in exps.items()}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def normalize_eval_label(val) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s in ("true", "yes", "1", "t", "y", "是"):
        return "true"
    if s in ("false", "no", "0", "f", "n", "否"):
        return "false"
    return s


def percentile_val(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * p))))
    return round(sorted_vals[idx], 2)


def evaluate_text_suite_for_model(m_cfg: dict, model_path: str) -> dict:
    alias = m_cfg["alias"]
    print(f"\n>>> [Text Suite] Warming up & loading model: {alias} ({model_path})...")

    # 1. Warmup via decision-cli noun --no-cache (ensures daemon + status bar are running and model is loaded in RAM)
    subprocess.run(
        [
            CLI_BIN, "noun",
            "--model", alias,
            "--rubrics", "判断是否为故障反馈",
            "--context", "系统启动崩溃报错",
            "--no-cache",
            "--format", "json",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    mon = HardwareMonitor(interval_sec=0.10)
    mon.start()
    t0_wall = time.perf_counter()

    # ---------------------------------------------------------------------
    # Part A: Chinese Mini-Bench (12 items) via `decision-cli batch noun --no-cache`
    # ---------------------------------------------------------------------
    mini_cases = get_mini_bench_data()
    by_rubric = defaultdict(list)
    for c in mini_cases:
        by_rubric[c["rubrics"]].append(c)

    mini_correct = 0
    mini_latencies = []
    for rub, group in by_rubric.items():
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
            json.dump([{"id": x["id"], "context": x["context"]} for x in group], tf, ensure_ascii=False)
            tpath = tf.name
        try:
            out = subprocess.check_output(
                [
                    CLI_BIN, "batch", "noun",
                    "--model", alias,
                    "--rubrics", rub,
                    "--input", tpath,
                    "--no-cache",
                    "--format", "json",
                ],
                text=True,
            )
            rep = json.loads(out)
            res_map = {r["id"]: r for r in rep["results"]}
            for x in group:
                r = res_map[x["id"]]
                if r["decision"] == x["expected"]:
                    mini_correct += 1
                mini_latencies.append(float(r.get("latency_ms", 0.0)))
        finally:
            if os.path.exists(tpath):
                os.remove(tpath)

    mini_acc = round(mini_correct / len(mini_cases), 4)

    # ---------------------------------------------------------------------
    # Part B: Path A 6-Class Choice (15 cases x Forward & Reverse) & 5-Level Score (10 cases)
    #         via `decision-cli batch choose/score --no-cache`
    # ---------------------------------------------------------------------
    path_a_data = get_path_a_benchmark_data()
    c_cases = path_a_data["choice_task"]["cases"]
    c_rubric = path_a_data["choice_task"]["rubrics"]
    c_list = path_a_data["choice_task"]["choices"]
    fwd_choices_str = ",".join(f'{c["key"]}:{c["description"]}' for c in c_list)
    rev_choices_str = ",".join(f'{c["key"]}:{c["description"]}' for c in reversed(c_list))

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
        json.dump([{"id": x["id"], "context": x["context"]} for x in c_cases], tf, ensure_ascii=False)
        c_path = tf.name
    try:
        out_fwd = json.loads(
            subprocess.check_output(
                [
                    CLI_BIN, "batch", "choose",
                    "--model", alias,
                    "--rubrics", c_rubric,
                    "--choices", fwd_choices_str,
                    "--input", c_path,
                    "--no-cache",
                    "--format", "json",
                ],
                text=True,
            )
        )
        out_rev = json.loads(
            subprocess.check_output(
                [
                    CLI_BIN, "batch", "choose",
                    "--model", alias,
                    "--rubrics", c_rubric,
                    "--choices", rev_choices_str,
                    "--input", c_path,
                    "--no-cache",
                    "--format", "json",
                ],
                text=True,
            )
        )
    finally:
        if os.path.exists(c_path):
            os.remove(c_path)

    fwd_map = {r["id"]: r for r in out_fwd["results"]}
    rev_map = {r["id"]: r for r in out_rev["results"]}
    fwd_corr = 0
    rev_corr = 0
    consist_cnt = 0
    choice_latencies = []
    for x in c_cases:
        rf = fwd_map[x["id"]]
        rr = rev_map[x["id"]]
        choice_latencies.append(float(rf.get("latency_ms", 0.0)))
        choice_latencies.append(float(rr.get("latency_ms", 0.0)))
        if rf["choice"] == x["expected"]:
            fwd_corr += 1
        if rr["choice"] == x["expected"]:
            rev_corr += 1
        if rf["choice"] == rr["choice"]:
            consist_cnt += 1

    s_cases = path_a_data["score_task"]["cases"]
    s_rubric = path_a_data["score_task"]["rubrics"]
    s_levels_str = ",".join(path_a_data["score_task"]["score_levels"])
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
        json.dump([{"id": x["id"], "context": x["context"]} for x in s_cases], tf, ensure_ascii=False)
        s_path = tf.name
    try:
        out_score = json.loads(
            subprocess.check_output(
                [
                    CLI_BIN, "batch", "score",
                    "--model", alias,
                    "--rubrics", s_rubric,
                    "--levels", s_levels_str,
                    "--input", s_path,
                    "--no-cache",
                    "--format", "json",
                ],
                text=True,
            )
        )
    finally:
        if os.path.exists(s_path):
            os.remove(s_path)

    s_map = {r["id"]: r for r in out_score["results"]}
    s_exact = 0
    s_mae_sum = 0.0
    score_latencies = []
    for x in s_cases:
        rs = s_map[x["id"]]
        score_latencies.append(float(rs.get("latency_ms", 0.0)))
        exp_score = float(rs.get("score", 0.0))
        pred_lvl = int(round(exp_score))
        if pred_lvl == x["expected_level"]:
            s_exact += 1
        s_mae_sum += abs(exp_score - float(x["expected_level"]))

    # ---------------------------------------------------------------------
    # Part C: JevBench Public 231 items (Easy 48 / Standard 72 / Hard 111)
    # ---------------------------------------------------------------------
    tiers = {
        "easy": load_jsonl(str(DATA_DIR / "easy.jsonl")),
        "standard": load_jsonl(str(DATA_DIR / "original.jsonl")),
        "hard": load_jsonl(str(DATA_DIR / "hard.jsonl")),
    }
    worker_items = []
    task_meta = {}
    for tier_name, t_list in tiers.items():
        for t in t_list:
            q = t.question
            qtype = q["type"]
            instr = q["instructions"]
            crit = q.get("criteria")
            ctx = t.state if isinstance(t.state, str) else json.dumps(t.state, ensure_ascii=False)
            if qtype == "noul":
                rubrics = instr
                if isinstance(crit, dict) and crit:
                    parts = []
                    for k, lbl in [("true", "YES (true)"), ("yes", "YES (true)"), ("false", "NO (false)"), ("no", "NO (false)")]:
                        if k in crit and crit[k]:
                            parts.append(f"- {lbl}: {crit[k]}")
                    if parts:
                        rubrics = instr.strip() + "\n" + "\n".join(parts)
                w_item = {"id": t.id, "primitive": "noul", "rubrics": rubrics, "context": ctx}
            elif qtype == "choice":
                choices = [{"key": lbl, "description": str(crit.get(lbl, lbl) if isinstance(crit, dict) else lbl)} for lbl in t.labels]
                w_item = {"id": t.id, "primitive": "choice", "rubrics": instr, "choices": choices, "context": ctx}
            elif qtype == "score":
                levels = [str(x) for x in crit] if isinstance(crit, list) else [str(x) for x in t.labels]
                w_item = {"id": t.id, "primitive": "score", "rubrics": instr, "score_levels": levels, "context": ctx}
            worker_items.append(w_item)
            task_meta[t.id] = (tier_name, t)

    resp_231 = send_worker_batch({
        "action": "score",
        "model_path": model_path,
        "prompt_style": m_cfg["prompt_style"],
        "positive_tokens": m_cfg["pos_tokens"],
        "negative_tokens": m_cfg["neg_tokens"],
        "debias_order": False,
        "items": worker_items,
    })
    if not resp_231.get("ok"):
        raise RuntimeError(f"JevBench 231 failed for {alias}: {resp_231.get('error')}")

    tier_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    jev_latencies = []

    for r in resp_231["results"]:
        tid = r["id"]
        tier_name, t = task_meta[tid]
        qtype = t.question["type"]
        gold = t.expected
        lat = float(r.get("latency_ms", 0.0))
        jev_latencies.append(lat)

        pred = None
        if qtype == "noul":
            margin = float(r.get("margin", 0.0))
            prob = sigmoid(margin / m_cfg["t_noul"])
            pred = "true" if prob >= 0.5 else "false"
        elif qtype == "choice":
            probs = softmax_dict(r.get("option_logits", {}), m_cfg["t_choice"])
            pred = max(probs.items(), key=lambda kv: kv[1])[0]
        elif qtype == "score":
            probs = softmax_dict(r.get("option_logits", {}), m_cfg["t_score"])
            exp_s = sum(int(k) * p for k, p in probs.items())
            pred = str(int(round(exp_s)))

        is_corr = normalize_eval_label(pred) == normalize_eval_label(gold)
        tier_stats[tier_name]["total"] += 1
        type_stats[qtype]["total"] += 1
        if is_corr:
            tier_stats[tier_name]["correct"] += 1
            type_stats[qtype]["correct"] += 1

    wall_elapsed_s = time.perf_counter() - t0_wall
    hw_usage = mon.stop()

    all_lats = sorted(mini_latencies + choice_latencies + score_latencies + jev_latencies)
    avg_lat = round(sum(all_lats) / len(all_lats), 2) if all_lats else 0.0
    p50_lat = percentile_val(all_lats, 0.50)
    p95_lat = percentile_val(all_lats, 0.95)
    dps = round(1000.0 / avg_lat, 2) if avg_lat > 0 else 0.0

    tot_jev_corr = sum(v["correct"] for v in tier_stats.values())
    tot_jev_cnt = sum(v["total"] for v in tier_stats.values())

    print(
        f"  [Done Text {alias}] JevBench={tot_jev_corr}/{tot_jev_cnt} ({tot_jev_corr/tot_jev_cnt*100:.2f}%), "
        f"Mini={mini_acc*100:.1f}%, AvgLat={avg_lat}ms ({dps} d/s), "
        f"Mem(Avg/Peak)={hw_usage['memory']['total_avg_mb']}/{hw_usage['memory']['total_peak_mb']} MB, "
        f"GPU(Avg/Peak)={hw_usage['gpu']['device_avg_pct']}/{hw_usage['gpu']['device_peak_pct']}%"
    )

    return {
        "total_text_decisions": len(all_lats),
        "wall_time_s": round(wall_elapsed_s, 2),
        "mini_bench": {
            "total": len(mini_cases),
            "correct": mini_correct,
            "accuracy": mini_acc,
        },
        "path_a": {
            "choice_forward_acc": round(fwd_corr / len(c_cases), 4),
            "choice_reverse_acc": round(rev_corr / len(c_cases), 4),
            "choice_order_consistency": round(consist_cnt / len(c_cases), 4),
            "score_exact_acc": round(s_exact / len(s_cases), 4),
            "score_mae": round(s_mae_sum / len(s_cases), 4),
        },
        "jevbench_231": {
            "total": tot_jev_cnt,
            "correct": tot_jev_corr,
            "overall_acc": round(tot_jev_corr / tot_jev_cnt, 4),
            "easy_acc": round(tier_stats["easy"]["correct"] / tier_stats["easy"]["total"], 4),
            "standard_acc": round(tier_stats["standard"]["correct"] / tier_stats["standard"]["total"], 4),
            "hard_acc": round(tier_stats["hard"]["correct"] / tier_stats["hard"]["total"], 4),
            "noul_acc": round(type_stats["noul"]["correct"] / type_stats["noul"]["total"], 4),
            "choice_acc": round(type_stats["choice"]["correct"] / type_stats["choice"]["total"], 4),
            "score_acc": round(type_stats["score"]["correct"] / type_stats["score"]["total"], 4),
        },
        "speed": {
            "avg_ms": avg_lat,
            "p50_ms": p50_lat,
            "p95_ms": p95_lat,
            "throughput_dps": dps,
        },
        "hardware_usage": hw_usage,
    }


def evaluate_multimodal_suite_for_model(m_cfg: dict, model_path: str, full_900_hist: dict) -> dict:
    alias = m_cfg["alias"]
    print(f"\n>>> [Multimodal Suite] Evaluating Vision + Prefill on {alias}...")

    meta_path = BOX_DIR / "items.jsonl"
    with open(meta_path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    # Stratified sample of 24 images (12 coco, 6 doc, 6 ui) x 3 tracks = 72 live uncached multimodal inferences
    by_fam = defaultdict(list)
    for it in all_items:
        by_fam[it["family"]].append(it)
    sampled_images = by_fam["coco"][:12] + by_fam["doc"][:6] + by_fam["ui"][:6]

    tracks = build_multimodal_benchmark(sampled_images)
    worker_items = []
    meta_map = {}

    for tname in ["track_1_noul", "track_2_classify", "track_3_grounding"]:
        for it in tracks[tname]:
            w_item = {
                "id": it["id"],
                "primitive": it["primitive"],
                "rubrics": it["rubrics"],
                "context": it["context"],
                "images": [it["image"]],
                "image_tokens": 280,
            }
            if it["primitive"] == "choice":
                w_item["choices"] = it["choices"]
            worker_items.append(w_item)
            meta_map[it["id"]] = (tname, it)

    mon = HardwareMonitor(interval_sec=0.10)
    mon.start()
    t0_wall = time.perf_counter()

    resp_mm = send_worker_batch({
        "action": "score",
        "model_path": model_path,
        "prompt_style": m_cfg["prompt_style"],
        "positive_tokens": m_cfg["pos_tokens"],
        "negative_tokens": m_cfg["neg_tokens"],
        "image_tokens": 280,
        "items": worker_items,
    })
    if not resp_mm.get("ok"):
        raise RuntimeError(f"Multimodal eval failed for {alias}: {resp_mm.get('error')}")

    wall_elapsed_s = time.perf_counter() - t0_wall
    hw_usage = mon.stop()

    track_stats = defaultdict(lambda: {"total": 0, "correct": 0, "lats": []})
    for r in resp_mm["results"]:
        iid = r["id"]
        tname, it = meta_map[iid]
        lat = float(r.get("latency_ms", 0.0))
        gold = str(it["expected"])

        if it["primitive"] == "noul":
            margin = float(r.get("margin", 0.0))
            prob = sigmoid(margin / m_cfg["t_noul"])
            pred = "true" if prob >= 0.5 else "false"
            is_corr = pred == gold.lower()
        else:
            probs = softmax_dict(r.get("option_logits", {}), m_cfg["t_choice"])
            pred = max(probs.items(), key=lambda kv: kv[1])[0]
            is_corr = pred.upper() == gold.upper()

        track_stats[tname]["total"] += 1
        track_stats[tname]["lats"].append(lat)
        if is_corr:
            track_stats[tname]["correct"] += 1

    # Track 1 encodes cold images through Vision Tower + Prefill; Tracks 2 & 3 hit Vision Feature LRU Cache + Prefill
    cold_lats = sorted(track_stats["track_1_noul"]["lats"])
    warm_lats = sorted(track_stats["track_2_classify"]["lats"] + track_stats["track_3_grounding"]["lats"])
    all_mm_lats = sorted(cold_lats + warm_lats)

    cold_avg = round(sum(cold_lats) / len(cold_lats), 2) if cold_lats else 0.0
    warm_avg = round(sum(warm_lats) / len(warm_lats), 2) if warm_lats else 0.0
    all_avg = round(sum(all_mm_lats) / len(all_mm_lats), 2) if all_mm_lats else 0.0

    # Combine with full 900-item benchmark accuracy for complete reporting
    h_model = full_900_hist.get("models", {}).get(alias, {})
    h_tracks = h_model.get("tracks", {})
    t1_900 = h_tracks.get("noul_verification", {}).get("accuracy", round(track_stats["track_1_noul"]["correct"] / track_stats["track_1_noul"]["total"], 4))
    t2_900 = h_tracks.get("choice_classification", {}).get("accuracy", round(track_stats["track_2_classify"]["correct"] / track_stats["track_2_classify"]["total"], 4))
    t3_900 = h_tracks.get("choice_spatial_grounding", {}).get("accuracy", round(track_stats["track_3_grounding"]["correct"] / track_stats["track_3_grounding"]["total"], 4))
    overall_900 = round((t1_900 + t2_900 + t3_900) / 3.0, 4)

    live_corr = sum(v["correct"] for v in track_stats.values())
    live_tot = sum(v["total"] for v in track_stats.values())

    print(
        f"  [Done Multimodal {alias}] Live(72)={live_corr}/{live_tot} ({live_corr/live_tot*100:.1f}%), "
        f"Full(900)={overall_900*100:.2f}%, ColdVisLat={cold_avg}ms, WarmPrefillLat={warm_avg}ms, "
        f"Mem(Avg/Peak)={hw_usage['memory']['total_avg_mb']}/{hw_usage['memory']['total_peak_mb']} MB, "
        f"GPU(Avg/Peak)={hw_usage['gpu']['device_avg_pct']}/{hw_usage['gpu']['device_peak_pct']}%"
    )

    return {
        "live_items": live_tot,
        "live_accuracy": round(live_corr / live_tot, 4),
        "wall_time_s": round(wall_elapsed_s, 2),
        "full_900_scores": {
            "overall_acc": overall_900,
            "track1_verify_acc": round(t1_900, 4),
            "track2_classify_acc": round(t2_900, 4),
            "track3_grounding_acc": round(t3_900, 4),
            "capability_score": h_model.get("macro_summary", {}).get("macro_capability_score", 0.0),
        },
        "speed": {
            "cold_vision_avg_ms": cold_avg,
            "cold_vision_p50_ms": percentile_val(cold_lats, 0.50),
            "warm_cache_avg_ms": warm_avg,
            "warm_cache_p50_ms": percentile_val(warm_lats, 0.50),
            "overall_avg_ms": all_avg,
            "overall_p50_ms": percentile_val(all_mm_lats, 0.50),
            "overall_p95_ms": percentile_val(all_mm_lats, 0.95),
            "throughput_dps": round(1000.0 / all_avg, 2) if all_avg > 0 else 0.0,
        },
        "hardware_usage": hw_usage,
    }


def main():
    full_900_path = WORKSPACE / "bench" / "results" / "jevbench_multimodal_900_eval_results.json"
    full_900_hist = {}
    if full_900_path.exists():
        with open(full_900_path, "r", encoding="utf-8") as f:
            full_900_hist = json.load(f)

    results = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Apple M5 Pro (15-Core CPU, Metal 4 GPU, 48 GB Unified Memory)",
        "runtime": "MLX 0.32.4 / mlx-lm 0.32.2 (Single-step Prefill Logit Reader)",
        "local_models": {},
        "remote_models": {
            "typesafe-jev-1.13.0": {
                "name": "TypeSafe Jev 1.13.0 (官方云端标杆)",
                "arch": "TypeSafe 官方 API / ImageJev Imajev-4B 参考",
                "supports_vision": True,
                "jevbench_231": {
                    "overall_acc": 0.8701,
                    "easy_acc": 1.0000,
                    "standard_acc": 0.9900,
                    "hard_acc": 0.7410,
                },
                "multimodal_ref": {
                    "model_note": "ImageJevBench v0.1.3 #1 Imajev-4B / #15 Jevify-26B",
                    "imajev_4b_open_acc": 0.8114,
                    "jevify_26b_open_acc": 0.8289,
                },
                "speed": {
                    "avg_ms": 285.0,
                    "p50_ms": 243.0,
                    "p95_ms": 620.0,
                    "throughput_dps": round(1000.0 / 285.0, 2),
                },
            },
        },
    }

    for m_cfg in LOCAL_MODELS:
        alias = m_cfg["alias"]
        m_path = resolve_local_model_path(alias)
        text_res = evaluate_text_suite_for_model(m_cfg, m_path)
        mm_res = None
        if m_cfg["supports_vision"]:
            mm_res = evaluate_multimodal_suite_for_model(m_cfg, m_path, full_900_hist)

        # Combine text & multimodal hardware usage for overall model resource summary
        t_mem = text_res["hardware_usage"]["memory"]
        t_gpu = text_res["hardware_usage"]["gpu"]
        if mm_res:
            m_mem = mm_res["hardware_usage"]["memory"]
            m_gpu = mm_res["hardware_usage"]["gpu"]
            comb_mem = {
                "worker_avg_mb": round((t_mem["worker_avg_mb"] + m_mem["worker_avg_mb"]) / 2, 1),
                "worker_peak_mb": max(t_mem["worker_peak_mb"], m_mem["worker_peak_mb"]),
                "bar_avg_mb": round((t_mem["bar_avg_mb"] + m_mem["bar_avg_mb"]) / 2, 1),
                "bar_peak_mb": max(t_mem["bar_peak_mb"], m_mem["bar_peak_mb"]),
                "cli_avg_mb": round((t_mem["cli_avg_mb"] + m_mem["cli_avg_mb"]) / 2, 1),
                "cli_peak_mb": max(t_mem["cli_peak_mb"], m_mem["cli_peak_mb"]),
                "total_avg_mb": round((t_mem["total_avg_mb"] + m_mem["total_avg_mb"]) / 2, 1),
                "total_peak_mb": max(t_mem["total_peak_mb"], m_mem["total_peak_mb"]),
            }
            comb_gpu = {
                "text_avg_pct": t_gpu["device_avg_pct"],
                "text_peak_pct": t_gpu["device_peak_pct"],
                "mm_avg_pct": m_gpu["device_avg_pct"],
                "mm_peak_pct": m_gpu["device_peak_pct"],
                "overall_avg_pct": round((t_gpu["device_avg_pct"] + m_gpu["device_avg_pct"]) / 2, 1),
                "overall_peak_pct": max(t_gpu["device_peak_pct"], m_gpu["device_peak_pct"]),
            }
        else:
            comb_mem = t_mem
            comb_gpu = {
                "text_avg_pct": t_gpu["device_avg_pct"],
                "text_peak_pct": t_gpu["device_peak_pct"],
                "mm_avg_pct": None,
                "mm_peak_pct": None,
                "overall_avg_pct": t_gpu["device_avg_pct"],
                "overall_peak_pct": t_gpu["device_peak_pct"],
            }

        results["local_models"][alias] = {
            "alias": alias,
            "arch": m_cfg["arch"],
            "supports_vision": m_cfg["supports_vision"],
            "text": text_res,
            "multimodal": mm_res,
            "resource_summary": {
                "memory": comb_mem,
                "gpu": comb_gpu,
            },
        }

    out_json = WORKSPACE / "bench" / "results" / "full_eval_results_20261030.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ All evaluations & hardware usage saved to: {out_json}")


if __name__ == "__main__":
    main()
