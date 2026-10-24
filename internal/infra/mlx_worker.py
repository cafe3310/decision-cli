#!/usr/bin/env python3
"""
mlx_worker.py — Apple Silicon MLX Non-Autoregressive Rubric Decision Worker
Supports:
  1. Single-pass prefill logit extraction (LogSumExp over positive_tokens vs negative_tokens)
  2. Unix Domain Socket resident daemon (--socket <path> --idle-timeout <sec>)
  3. One-shot stdin/stdout JSON-RPC mode (--oneshot)
"""

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback

import mlx.core as mx
from mlx_lm import load


def logsumexp(vals):
    if not vals:
        return -100.0
    m = max(vals)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def build_prompt(tokenizer, prompt_style: str, rubrics: str, context: str) -> str:
    if prompt_style == "qwen3-reranker":
        return (
            "<|im_start|>system\n"
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
            f"<Instruct>: {rubrics}\n"
            f"<Query>: {rubrics}\n"
            f"<Document>: {context}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
    else:
        # gemma4-binary or generic instruction-tuned binary judge
        user_msg = (
            "你是一个精确的文本分类与判别器。请严格根据【判定准则 (Rubric)】判断【待测文本 (Context)】是否符合要求。\n"
            "仅回答一个字：“是”或“否”，不要输出任何解释或标点符号。\n\n"
            f"【判定准则 (Rubric)】\n{rubrics}\n\n"
            f"【待测文本 (Context)】\n{context}\n\n"
            "【判定结果（仅回答“是”或“否”）】"
        )
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": user_msg}]
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                try:
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
        return (
            f"<bos><start_of_turn>user\n{user_msg}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )


def resolve_candidate_token_ids(tokenizer, token_strs):
    ids = set()
    for t in token_strs:
        for variant in (t, " " + t):
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
                elif len(enc) > 1:
                    # Take first content token if single-token encoding isn't direct
                    ids.add(enc[0])
            except Exception:
                pass
    return sorted(ids)


class ModelCache:
    def __init__(self):
        self.loaded_path = None
        self.model = None
        self.tokenizer = None
        self.language_model = None

    def get(self, model_path: str):
        if self.loaded_path != model_path or self.model is None:
            t0 = time.perf_counter()
            model, tokenizer = load(model_path)
            # Some VLM wrappers (e.g., Gemma-4 VLM architecture in mlx_lm) expose language_model
            self.model = model
            self.tokenizer = tokenizer
            self.loaded_path = model_path
        return self.model, self.tokenizer


MODEL_CACHE = ModelCache()


def forward_last_logits(model, input_ids):
    """Run a single prefill forward pass and return the 1D log-probabilities or logits at position -1."""
    x = mx.array([input_ids], dtype=mx.int32)
    try:
        out = model(x)
    except Exception:
        if hasattr(model, "language_model"):
            out = model.language_model(x)
        else:
            raise
    if isinstance(out, tuple):
        out = out[0]
    elif hasattr(out, "logits"):
        out = out.logits
    last_logits = out[0, -1, :].astype(mx.float32)
    # Convert raw logits to log-probabilities over vocabulary for numerical consistency
    log_probs = last_logits - mx.logsumexp(last_logits)
    mx.eval(log_probs)
    return log_probs


def handle_score_request(req: dict, stream_stdout: bool = False) -> dict:
    model_path = req["model_path"]
    prompt_style = req.get("prompt_style", "gemma4-binary")
    rubrics = req["rubrics"]
    pos_tokens = req.get("positive_tokens") or ["是", "Yes", "yes", "YES"]
    neg_tokens = req.get("negative_tokens") or ["否", "No", "no", "NO"]
    items = req.get("items", [])

    if stream_stdout:
        sys.stdout.write(json.dumps({"type": "status", "status": "loading_model", "model_path": model_path}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    model, tokenizer = MODEL_CACHE.get(model_path)
    pos_ids = resolve_candidate_token_ids(tokenizer, pos_tokens)
    neg_ids = resolve_candidate_token_ids(tokenizer, neg_tokens)

    if not pos_ids or not neg_ids:
        raise ValueError(f"Failed to resolve token IDs for pos={pos_tokens} ({pos_ids}) or neg={neg_tokens} ({neg_ids})")

    if stream_stdout:
        sys.stdout.write(json.dumps({"type": "status", "status": "scoring", "total": len(items)}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    results = []
    for item in items:
        t0 = time.perf_counter()
        item_id = item.get("id", "")
        context = item.get("context", "")
        prompt = build_prompt(tokenizer, prompt_style, rubrics, context)
        input_ids = tokenizer.encode(prompt)
        log_probs = forward_last_logits(model, input_ids)

        pos_vals = [float(log_probs[idx].item()) for idx in pos_ids]
        neg_vals = [float(log_probs[idx].item()) for idx in neg_ids]
        logit_yes = logsumexp(pos_vals)
        logit_no = logsumexp(neg_vals)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        res_obj = {
            "id": item_id,
            "logit_yes": round(logit_yes, 6),
            "logit_no": round(logit_no, 6),
            "margin": round(logit_yes - logit_no, 6),
            "latency_ms": round(latency_ms, 2),
        }
        results.append(res_obj)
        if stream_stdout:
            sys.stdout.write(json.dumps({"type": "item", "result": res_obj}, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    return {
        "ok": True,
        "type": "done",
        "model_path": model_path,
        "pos_token_ids": pos_ids,
        "neg_token_ids": neg_ids,
        "results": results,
    }


def dispatch_request(req: dict, stream_stdout: bool = False) -> dict:
    action = req.get("action", "score")
    if action == "ping":
        return {
            "ok": True,
            "loaded_model": MODEL_CACHE.loaded_path,
            "pid": os.getpid(),
        }
    elif action == "score":
        return handle_score_request(req, stream_stdout=stream_stdout)
    elif action == "shutdown":
        return {"ok": True, "shutdown": True}
    else:
        return {"ok": False, "error": f"unknown action: {action}"}


def serve_socket(sock_path: str, idle_timeout: float):
    if os.path.exists(sock_path):
        try:
            os.unlink(sock_path)
        except OSError:
            pass
    os.makedirs(os.path.dirname(sock_path), exist_ok=True)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(8)
    srv.settimeout(1.0)

    last_active = time.time()
    should_exit = False

    try:
        while not should_exit:
            if idle_timeout > 0 and (time.time() - last_active) > idle_timeout:
                break
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue

            last_active = time.time()
            with conn:
                data = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in chunk:
                        break
                if not data:
                    continue
                try:
                    req = json.loads(data.decode("utf-8").strip())
                    resp = dispatch_request(req)
                    if resp.get("shutdown"):
                        should_exit = True
                except Exception as e:
                    resp = {
                        "ok": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                payload = (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8")
                conn.sendall(payload)
            last_active = time.time()
    finally:
        srv.close()
        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="decision-cli MLX worker")
    parser.add_argument("--socket", type=str, default="", help="Unix domain socket path")
    parser.add_argument("--idle-timeout", type=float, default=300.0, help="Idle timeout in seconds")
    parser.add_argument("--oneshot", action="store_true", help="Read single JSON request from stdin and write to stdout")
    args = parser.parse_args()

    if args.oneshot:
        raw = sys.stdin.read()
        try:
            req = json.loads(raw)
            resp = dispatch_request(req, stream_stdout=bool(req.get("stream")))
        except Exception as e:
            resp = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return

    if not args.socket:
        print("Error: either --socket or --oneshot must be specified", file=sys.stderr)
        sys.exit(1)

    serve_socket(args.socket, args.idle_timeout)


if __name__ == "__main__":
    main()
