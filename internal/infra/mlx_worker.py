#!/usr/bin/env python3
"""
mlx_worker.py — Apple Silicon MLX Non-Autoregressive Multimodal Rubric Decision Worker
Supports:
  1. Single-pass prefill logit extraction (LogSumExp over positive_tokens vs negative_tokens)
  2. Native Gemma 4 Vision Encoder (gemma-4-26b-a4b & gemma-4-e4b) with single/multi-image support
     and in-memory LRU vision feature caching
  3. Unix Domain Socket resident daemon (--socket <path> --idle-timeout <sec>)
  4. One-shot stdin/stdout JSON-RPC mode (--oneshot)
"""

import argparse
import base64
from collections import OrderedDict
import hashlib
import io
import json
import math
import os
import re
import socket
import sys
import time
import traceback

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from mlx_lm import load


SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
GEMMA4_BOI_TOKEN = "<|image>"
GEMMA4_IMG_TOKEN = "<|image|>"
GEMMA4_EOI_TOKEN = "<image|>"
GEMMA4_IMG_TOKEN_ID = 258880


def logsumexp(vals):
    if not vals:
        return -100.0
    m = max(vals)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


TAG_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def normalize_soft_tokens(val: int) -> int:
    if not val or val <= 0:
        return 280
    if val in SUPPORTED_SOFT_TOKENS:
        return val
    # Snap to nearest supported Gemma 4 soft token budget
    return min(SUPPORTED_SOFT_TOKENS, key=lambda x: abs(x - val))


def load_image_and_hash(img_spec: str) -> tuple[Image.Image, str]:
    spec = img_spec.strip()
    if spec.startswith("data:image/"):
        comma_idx = spec.find(",")
        if comma_idx == -1:
            raise ValueError("invalid data:image URI: missing comma separator")
        raw_bytes = base64.b64decode(spec[comma_idx + 1 :])
    elif spec.startswith("base64:"):
        raw_bytes = base64.b64decode(spec[len("base64:") :])
    else:
        path = spec
        if path.startswith("file://"):
            path = path[len("file://") :]
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                raw_bytes = f.read()
        else:
            # Try base64 fallback if it looks like raw base64 and file doesn't exist
            if len(spec) > 128 and not os.path.sep in spec[:32]:
                try:
                    raw_bytes = base64.b64decode(spec, validate=True)
                except Exception as exc:
                    raise FileNotFoundError(f"image file not found: {path}") from exc
            else:
                raise FileNotFoundError(f"image file not found: {path}")

    digest = hashlib.sha256(raw_bytes).hexdigest()
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return img, digest


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int = 16,
    max_patches: int = 2520,
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    max_side_length = (max_patches // (pooling_kernel_size ** 2)) * side_mult
    if target_height == 0 and target_width == 0:
        target_height = side_mult
        target_width = side_mult
    elif target_height == 0:
        target_height = side_mult
        target_width = min(max(1, int(math.floor(width / height))) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(max(1, int(math.floor(height / width))) * side_mult, max_side_length)

    return target_height, target_width


def preprocess_gemma4_image(img: Image.Image, max_soft_tokens: int = 280):
    max_soft_tokens = normalize_soft_tokens(max_soft_tokens)
    orig_w, orig_h = img.size
    patch_size = 16
    pooling_kernel_size = 3
    max_patches = max_soft_tokens * (pooling_kernel_size ** 2)
    target_h, target_w = get_aspect_ratio_preserving_size(
        orig_h, orig_w, patch_size, max_patches, pooling_kernel_size
    )
    resized = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    hp, wp = target_h // patch_size, target_w // patch_size
    num_patches = hp * wp
    patches = (
        arr.reshape(hp, patch_size, wp, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(1, num_patches, 768)
    )
    gx, gy = np.meshgrid(np.arange(wp, dtype=np.int32), np.arange(hp, dtype=np.int32), indexing="xy")
    pos_ids = np.stack([gx, gy], axis=-1).reshape(1, num_patches, 2)
    return mx.array(patches), mx.array(pos_ids), hp, wp, num_patches // 9


def rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multidim_rope(x, cos, sin):
    x_parts = mx.split(x, 2, axis=-1)
    cos_parts = mx.split(cos[:, :, None, :], 2, axis=-1)
    sin_parts = mx.split(sin[:, :, None, :], 2, axis=-1)
    out = [xp * cp + rotate_half(xp) * sp for xp, cp, sp in zip(x_parts, cos_parts, sin_parts)]
    return mx.concatenate(out, axis=-1)


class Gemma4VisionRunner:
    """Native MLX Gemma 4 Vision Tower + Multimodal Embedder with LRU feature caching."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        if "vision_config" not in self.cfg:
            raise ValueError(f"Model at {model_dir} does not contain vision_config (text-only model)")

        self.vcfg = self.cfg["vision_config"]
        self.tcfg = self.cfg.get("text_config", {})
        self.hidden_size = self.vcfg["hidden_size"]
        self.num_layers = self.vcfg["num_hidden_layers"]
        self.num_heads = self.vcfg["num_attention_heads"]
        self.head_dim = self.vcfg.get("head_dim", self.hidden_size // self.num_heads)
        self.eps = self.vcfg.get("rms_norm_eps", 1e-6)
        self.standardize = bool(self.vcfg.get("standardize", False))
        self.use_clipped = bool(self.vcfg.get("use_clipped_linears", False))
        self.rope_theta = float(self.vcfg.get("rope_parameters", {}).get("rope_theta", 100.0))
        spatial_dim = self.head_dim // 2
        self.inv_freq = 1.0 / (
            self.rope_theta ** (mx.arange(0, spatial_dim, 2, dtype=mx.float32) / float(spatial_dim))
        )

        # Load vision_tower.* and embed_vision.* weights from safetensors
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                wmap = json.load(f).get("weight_map", {})
            v_shards = sorted(
                set(
                    sh
                    for k, sh in wmap.items()
                    if k.startswith("vision_tower.") or k.startswith("embed_vision.")
                )
            )
        else:
            v_shards = ["model.safetensors"]

        self.weights = {}
        for sh in v_shards:
            sh_path = os.path.join(model_dir, sh)
            loaded = mx.load(sh_path)
            for k, v in loaded.items():
                if k.startswith("vision_tower.") or k.startswith("embed_vision."):
                    self.weights[k] = v

        if "vision_tower.patch_embedder.input_proj.weight" not in self.weights:
            raise ValueError(f"Model at {model_dir} is missing vision_tower weights in safetensors")

        self.feature_cache = OrderedDict()
        self.max_cache_items = 128

    def _linear(self, x, prefix: str):
        if self.use_clipped and f"{prefix}.input_min" in self.weights:
            x = mx.clip(x, self.weights[f"{prefix}.input_min"], self.weights[f"{prefix}.input_max"])
        w = self.weights[f"{prefix}.linear.weight"]
        x = x @ w.T
        if self.use_clipped and f"{prefix}.output_min" in self.weights:
            x = mx.clip(x, self.weights[f"{prefix}.output_min"], self.weights[f"{prefix}.output_max"])
        return x

    def encode_image(self, img_spec: str, max_soft_tokens: int = 280) -> tuple[mx.array, int]:
        max_soft_tokens = normalize_soft_tokens(max_soft_tokens)
        img, digest = load_image_and_hash(img_spec)
        cache_key = (digest, max_soft_tokens)
        if cache_key in self.feature_cache:
            self.feature_cache.move_to_end(cache_key)
            return self.feature_cache[cache_key]

        pixel_values, pos_ids, hp, wp, n_soft = preprocess_gemma4_image(img, max_soft_tokens)
        w = self.weights
        proj_w = w["vision_tower.patch_embedder.input_proj.weight"]
        pv = (2.0 * (pixel_values - 0.5)).astype(proj_w.dtype)
        h = pv @ proj_w.T
        pos_tab = w["vision_tower.patch_embedder.position_embedding_table"]
        pos_emb = pos_tab[0][pos_ids[..., 0]] + pos_tab[1][pos_ids[..., 1]]
        h = h + pos_emb

        freqs = pos_ids[..., None].astype(mx.float32) * self.inv_freq
        cos_f, sin_f = mx.cos(freqs), mx.sin(freqs)
        cos = mx.concatenate(
            [cos_f[:, :, 0, :], cos_f[:, :, 0, :], cos_f[:, :, 1, :], cos_f[:, :, 1, :]], axis=-1
        ).astype(h.dtype)
        sin = mx.concatenate(
            [sin_f[:, :, 0, :], sin_f[:, :, 0, :], sin_f[:, :, 1, :], sin_f[:, :, 1, :]], axis=-1
        ).astype(h.dtype)

        B, L, D = h.shape
        for i in range(self.num_layers):
            lp = f"vision_tower.encoder.layers.{i}"
            normed = mx.fast.rms_norm(h, w[f"{lp}.input_layernorm.weight"], self.eps)
            q = self._linear(normed, f"{lp}.self_attn.q_proj").reshape(B, L, self.num_heads, self.head_dim)
            q = mx.fast.rms_norm(q, w[f"{lp}.self_attn.q_norm.weight"], self.eps)
            q = apply_multidim_rope(q, cos, sin).transpose(0, 2, 1, 3)

            k = self._linear(normed, f"{lp}.self_attn.k_proj").reshape(B, L, self.num_heads, self.head_dim)
            k = mx.fast.rms_norm(k, w[f"{lp}.self_attn.k_norm.weight"], self.eps)
            k = apply_multidim_rope(k, cos, sin).transpose(0, 2, 1, 3)

            v = self._linear(normed, f"{lp}.self_attn.v_proj").reshape(B, L, self.num_heads, self.head_dim)
            v = mx.fast.rms_norm(v, None, self.eps).transpose(0, 2, 1, 3)

            attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)
            attn_out = self._linear(attn_out, f"{lp}.self_attn.o_proj")
            attn_out = mx.fast.rms_norm(attn_out, w[f"{lp}.post_attention_layernorm.weight"], self.eps)
            h = h + attn_out

            mlp_in = mx.fast.rms_norm(h, w[f"{lp}.pre_feedforward_layernorm.weight"], self.eps)
            gate = nn.gelu_approx(self._linear(mlp_in, f"{lp}.mlp.gate_proj"))
            up = self._linear(mlp_in, f"{lp}.mlp.up_proj")
            mlp_out = self._linear(gate * up, f"{lp}.mlp.down_proj")
            mlp_out = mx.fast.rms_norm(mlp_out, w[f"{lp}.post_feedforward_layernorm.weight"], self.eps)
            h = h + mlp_out

        pooled = (
            h.astype(mx.float32)
            .reshape(1, hp // 3, 3, wp // 3, 3, D)
            .mean(axis=(2, 4))
            .reshape(1, n_soft, D)
        )
        pooled = pooled.astype(h.dtype).astype(mx.float32) * math.sqrt(D)
        if self.standardize:
            std_bias = w["vision_tower.std_bias"].astype(mx.float32)
            std_scale = w["vision_tower.std_scale"].astype(mx.float32)
            pooled = (pooled - std_bias) * std_scale
        pooled = pooled.astype(h.dtype)

        normed_pool = mx.fast.rms_norm(pooled, None, self.eps)
        if "embed_vision.embedding_projection.scales" in w:
            qcfg = self.cfg.get("quantization") or self.cfg.get("quantization_config") or {}
            img_embeds = mx.quantized_matmul(
                normed_pool,
                w["embed_vision.embedding_projection.weight"],
                scales=w["embed_vision.embedding_projection.scales"],
                biases=w.get("embed_vision.embedding_projection.biases"),
                transpose=True,
                group_size=int(qcfg.get("group_size", 64)),
                bits=int(qcfg.get("bits", 4)),
            )
        else:
            img_embeds = normed_pool @ w["embed_vision.embedding_projection.weight"].T

        mx.eval(img_embeds)
        res = (img_embeds[0], n_soft)
        self.feature_cache[cache_key] = res
        if len(self.feature_cache) > self.max_cache_items:
            self.feature_cache.popitem(last=False)
        return res


def expand_image_block(n_soft: int) -> str:
    return GEMMA4_BOI_TOKEN + (GEMMA4_IMG_TOKEN * n_soft) + GEMMA4_EOI_TOKEN


def inject_multimodal_placeholders(rubrics: str, context: str, n_soft_list: list[int]) -> tuple[str, str, list[int]]:
    """
    Replaces {{image:1}}, {{image:2}}, ..., {{image}}, or <|image|> in rubrics and context with expanded
    Gemma 4 image token blocks (<|image><|image|>...<image|>), and returns (rubrics, context, ordered_image_indices).
    Any image not explicitly referenced via placeholder is prepended to context in order.
    """
    if not n_soft_list:
        return rubrics, context, []

    used_indices = set()
    unnumbered_counter = [0]

    def replace_unnumbered(match):
        idx = unnumbered_counter[0]
        if idx < len(n_soft_list):
            unnumbered_counter[0] += 1
            used_indices.add(idx)
            return f"__DECISION_IMAGE_SLOT_{idx}__"
        return match.group(0)

    def replace_numbered(match):
        one_based = int(match.group(1))
        idx = one_based - 1
        if 0 <= idx < len(n_soft_list):
            used_indices.add(idx)
            return f"__DECISION_IMAGE_SLOT_{idx}__"
        return match.group(0)

    rubrics_s = re.sub(r"\{\{\s*image\s*:\s*(\d+)\s*\}\}", replace_numbered, rubrics)
    context_s = re.sub(r"\{\{\s*image\s*:\s*(\d+)\s*\}\}", replace_numbered, context)
    rubrics_s = re.sub(r"(?:\{\{\s*image\s*\}\}|<\|image\|>)", replace_unnumbered, rubrics_s)
    context_s = re.sub(r"(?:\{\{\s*image\s*\}\}|<\|image\|>)", replace_unnumbered, context_s)

    # Any image not explicitly placed is prepended to context_s
    prepended_slots = []
    for idx in range(len(n_soft_list)):
        if idx not in used_indices:
            if len(n_soft_list) > 1:
                prepended_slots.append(f"[Image {idx + 1}: __DECISION_IMAGE_SLOT_{idx}__]")
            else:
                prepended_slots.append(f"__DECISION_IMAGE_SLOT_{idx}__")

    if prepended_slots:
        prefix_str = "\n".join(prepended_slots)
        if context_s.strip():
            context_s = prefix_str + "\n" + context_s
        else:
            context_s = prefix_str

    # Determine final order of image slots as they appear in (rubrics_s, context_s)
    full_Order_scan = rubrics_s + "\n" + context_s
    ordered_indices = [int(m.group(1)) for m in re.finditer(r"__DECISION_IMAGE_SLOT_(\d+)__", full_Order_scan)]

    for idx, n_soft in enumerate(n_soft_list):
        block = expand_image_block(n_soft)
        rubrics_s = rubrics_s.replace(f"__DECISION_IMAGE_SLOT_{idx}__", block)
        context_s = context_s.replace(f"__DECISION_IMAGE_SLOT_{idx}__", block)

    return rubrics_s, context_s, ordered_indices


def apply_chat_or_raw(tokenizer, prompt_style: str, user_msg: str) -> str:
    if prompt_style == "qwen3-reranker":
        return (
            "<|im_start|>system\n"
            "You are a precise classification and decision assistant. Output only the requested option token.<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
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


def build_prompt(tokenizer, prompt_style: str, rubrics: str, context: str, has_images: bool = False) -> str:
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
    target_label = "【待测内容与图像 (Context)】" if has_images else "【待测文本 (Context)】"
    role_desc = "你是一个精确的多模态图文分类与判别器。" if has_images else "你是一个精确的文本分类与判别器。"
    user_msg = (
        f"{role_desc}请严格根据【判定准则 (Rubric)】判断{target_label}是否符合要求。\n"
        "仅回答一个字：“是”或“否”，不要输出任何解释或标点符号。\n\n"
        f"【判定准则 (Rubric)】\n{rubrics}\n\n"
        f"{target_label}\n{context}\n\n"
        "【判定结果（仅回答“是”或“否”）】"
    )
    return apply_chat_or_raw(tokenizer, prompt_style, user_msg)


def build_choice_prompt(tokenizer, prompt_style: str, rubrics: str, ordered_choices: list, context: str, has_images: bool = False) -> str:
    lines = []
    tags_used = []
    for idx, c in enumerate(ordered_choices):
        tag = TAG_POOL[idx]
        tags_used.append(tag)
        key = str(c.get("key", "")).strip()
        desc = str(c.get("description", "")).strip()
        if desc and desc != key:
            lines.append(f"{tag}: {key} — {desc}")
        else:
            lines.append(f"{tag}: {key}")
    options_block = "\n".join(lines)
    tag_hint = "、".join(tags_used[:min(4, len(tags_used))])
    target_label = "【待测内容与图像 (Context)】" if has_images else "【待测文本 (Context)】"
    user_msg = (
        f"你是一个精确的多分类决策器。请严格根据【分类指令 (Instructions)】与【候选类别 (Criteria)】的定义，"
        f"判断{target_label}最符合哪一个类别。\n"
        f"仅输出一个代表该类别的单字符代号（如 {tag_hint}），不要输出任何解释、空格或标点符号。\n\n"
        f"【分类指令 (Instructions)】\n{rubrics}\n\n"
        f"【候选类别 (Criteria)】\n{options_block}\n\n"
        f"{target_label}\n{context}\n\n"
        "【最佳类别代号（仅回答单个字符代号）】"
    )
    return apply_chat_or_raw(tokenizer, prompt_style, user_msg)


def build_score_prompt(tokenizer, prompt_style: str, rubrics: str, ordered_levels: list, context: str, has_images: bool = False) -> str:
    lines = []
    m_count = len(ordered_levels)
    for display_idx, (_, desc) in enumerate(ordered_levels):
        lines.append(f"{display_idx}: {desc}")
    levels_block = "\n".join(lines)
    target_label = "【待测内容与图像 (Context)】" if has_images else "【待测文本 (Context)】"
    user_msg = (
        f"你是一个精确的量规评分器。请严格对照【评分维度 (Instructions)】与【档位标准 (Scale)】，"
        f"对{target_label}进行打分。\n"
        f"仅输出一个代表档位的数字（0 到 {m_count - 1}），不要输出任何解释或标点符号。\n\n"
        f"【评分维度 (Instructions)】\n{rubrics}\n\n"
        f"【档位标准 (Scale)】\n{levels_block}\n\n"
        f"{target_label}\n{context}\n\n"
        f"【评分档位（仅回答单个数字 0-{m_count - 1}）】"
    )
    return apply_chat_or_raw(tokenizer, prompt_style, user_msg)


def resolve_candidate_token_ids(tokenizer, token_strs):
    ids = set()
    for t in token_strs:
        try:
            enc = tokenizer.encode(t, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
            elif len(enc) > 1:
                ids.add(enc[0])
        except Exception:
            pass
        try:
            enc_sp = tokenizer.encode(" " + t, add_special_tokens=False)
            if len(enc_sp) == 1:
                ids.add(enc_sp[0])
        except Exception:
            pass
    return sorted(ids)


class ModelCache:
    def __init__(self):
        self.loaded_path = None
        self.model = None
        self.tokenizer = None
        self.vision_runner = None

    def get(self, model_path: str):
        if self.loaded_path != model_path or self.model is None:
            model, tokenizer = load(model_path)
            self.model = model
            self.tokenizer = tokenizer
            self.loaded_path = model_path
            self.vision_runner = None
        return self.model, self.tokenizer

    def get_vision_runner(self, model_path: str) -> Gemma4VisionRunner:
        self.get(model_path)
        if self.vision_runner is None:
            self.vision_runner = Gemma4VisionRunner(model_path)
        return self.vision_runner


MODEL_CACHE = ModelCache()


def forward_last_logits(model, input_ids: list[int], ordered_img_embs: list[mx.array] | None = None):
    """Run a single prefill forward pass (text-only or multimodal) and return 1D log-probabilities at position -1."""
    lm = model.language_model if hasattr(model, "language_model") else model

    if not ordered_img_embs:
        x = mx.array([input_ids], dtype=mx.int32)
        try:
            out = model(x)
        except Exception:
            out = lm(x)
    else:
        ids_np = np.array(input_ids, dtype=np.int32)
        mask = ids_np == GEMMA4_IMG_TOKEN_ID
        llm_ids = mx.array(np.where(mask, 0, ids_np)[None, :])
        text_embeds = lm.model.embed_tokens(llm_ids)
        embed_scale = float(getattr(lm.model, "embed_scale", 1.0))

        # Locate contiguous runs of GEMMA4_IMG_TOKEN_ID
        padded = np.concatenate(([False], mask, [False]))
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])

        if len(starts) != len(ordered_img_embs):
            raise ValueError(
                f"Mismatch between image token spans ({len(starts)}) and encoded images ({len(ordered_img_embs)})"
            )

        segments = []
        prev = 0
        for (s, e), emb in zip(zip(starts, ends), ordered_img_embs):
            s_i, e_i = int(s), int(e)
            if s_i > prev:
                segments.append(text_embeds[:, prev:s_i, :])
            scaled_img = (emb / embed_scale).astype(text_embeds.dtype)[None, :, :]
            segments.append(scaled_img)
            prev = e_i
        if prev < len(input_ids):
            segments.append(text_embeds[:, prev:, :])

        merged_embeds = mx.concatenate(segments, axis=1)
        out = lm(inputs=llm_ids, input_embeddings=merged_embeds)

    if isinstance(out, tuple):
        out = out[0]
    elif hasattr(out, "logits"):
        out = out.logits
    last_logits = out[0, -1, :].astype(mx.float32)
    log_probs = last_logits - mx.logsumexp(last_logits)
    mx.eval(log_probs)
    return log_probs


def prepare_item_multimodal(
    model_path: str,
    rubrics: str,
    context: str,
    images: list[str],
    image_tokens: int,
) -> tuple[str, str, list[mx.array]]:
    if not images:
        return rubrics, context, []
    vrunner = MODEL_CACHE.get_vision_runner(model_path)
    encoded_pairs = [vrunner.encode_image(img_spec, image_tokens) for img_spec in images]
    embs = [p[0] for p in encoded_pairs]
    n_soft_list = [p[1] for p in encoded_pairs]
    rubrics_m, context_m, ordered_indices = inject_multimodal_placeholders(rubrics, context, n_soft_list)
    ordered_embs = [embs[idx] for idx in ordered_indices]
    return rubrics_m, context_m, ordered_embs


def score_single_item(
    model_path: str,
    model,
    tokenizer,
    prompt_style: str,
    default_prim: str,
    default_rubrics: str,
    default_choices: list,
    default_levels: list,
    default_debias: bool,
    default_image_tokens: int,
    pos_ids: list,
    neg_ids: list,
    item: dict,
) -> dict:
    t0 = time.perf_counter()
    item_id = item.get("id", "")
    context = item.get("context", "")
    prim = item.get("primitive") or default_prim
    rubrics = item.get("rubrics") or default_rubrics
    images = item.get("images") or []
    image_tokens = int(item.get("image_tokens") or default_image_tokens or 280)
    debias_order = (
        item["debias_order"]
        if "debias_order" in item and item["debias_order"] is not None
        else default_debias
    )

    rubrics_m, context_m, ordered_embs = prepare_item_multimodal(
        model_path, rubrics, context, images, image_tokens
    )
    has_images = len(ordered_embs) > 0

    if prim == "choice":
        choices = item.get("choices") or default_choices
        k_count = len(choices)
        if k_count < 2 or k_count > len(TAG_POOL):
            raise ValueError(f"choice primitive requires 2..{len(TAG_POOL)} options, got {k_count}")
        tag_token_ids = [resolve_candidate_token_ids(tokenizer, [TAG_POOL[i]]) for i in range(k_count)]

        def run_choice_pass(ordered_choices: list) -> dict:
            prompt = build_choice_prompt(
                tokenizer, prompt_style, rubrics_m, ordered_choices, context_m, has_images=has_images
            )
            input_ids = tokenizer.encode(prompt, add_special_tokens=False) if has_images else tokenizer.encode(prompt)
            log_probs = forward_last_logits(model, input_ids, ordered_embs)
            raw_vals = {}
            for idx, c in enumerate(ordered_choices):
                tids = tag_token_ids[idx]
                raw_vals[c["key"]] = logsumexp([float(log_probs[tid].item()) for tid in tids])
            sub_lse = logsumexp(list(raw_vals.values()))
            return {k: v - sub_lse for k, v in raw_vals.items()}

        pass1 = run_choice_pass(choices)
        if debias_order and k_count >= 2:
            pass2 = run_choice_pass(list(reversed(choices)))
            merged_raw = {k: 0.5 * (pass1[k] + pass2[k]) for k in pass1}
            sub_lse = logsumexp(list(merged_raw.values()))
            option_logits = {k: round(v - sub_lse, 6) for k, v in merged_raw.items()}
        else:
            option_logits = {k: round(v, 6) for k, v in pass1.items()}

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "id": item_id,
            "primitive": "choice",
            "option_logits": option_logits,
            "latency_ms": round(latency_ms, 2),
        }

    elif prim == "score":
        score_levels = item.get("score_levels") or default_levels
        m_count = len(score_levels)
        if m_count < 2 or m_count > 10:
            raise ValueError(f"score primitive requires 2..10 ordinal levels, got {m_count}")
        digit_token_ids = [resolve_candidate_token_ids(tokenizer, [str(m)]) for m in range(m_count)]
        indexed_levels = list(enumerate(score_levels))

        def run_score_pass(ordered_pairs: list) -> dict:
            prompt = build_score_prompt(
                tokenizer, prompt_style, rubrics_m, ordered_pairs, context_m, has_images=has_images
            )
            input_ids = tokenizer.encode(prompt, add_special_tokens=False) if has_images else tokenizer.encode(prompt)
            log_probs = forward_last_logits(model, input_ids, ordered_embs)
            raw_vals = {}
            for display_idx, (orig_idx, _) in enumerate(ordered_pairs):
                tids = digit_token_ids[display_idx]
                raw_vals[str(orig_idx)] = logsumexp([float(log_probs[tid].item()) for tid in tids])
            sub_lse = logsumexp(list(raw_vals.values()))
            return {k: v - sub_lse for k, v in raw_vals.items()}

        pass1 = run_score_pass(indexed_levels)
        if debias_order and m_count >= 2:
            pass2 = run_score_pass(list(reversed(indexed_levels)))
            merged_raw = {k: 0.5 * (pass1[k] + pass2[k]) for k in pass1}
            sub_lse = logsumexp(list(merged_raw.values()))
            option_logits = {k: round(v - sub_lse, 6) for k, v in merged_raw.items()}
        else:
            option_logits = {k: round(v, 6) for k, v in pass1.items()}

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "id": item_id,
            "primitive": "score",
            "option_logits": option_logits,
            "latency_ms": round(latency_ms, 2),
        }

    else:
        prompt = build_prompt(tokenizer, prompt_style, rubrics_m, context_m, has_images=has_images)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False) if has_images else tokenizer.encode(prompt)
        log_probs = forward_last_logits(model, input_ids, ordered_embs)
        pos_vals = [float(log_probs[idx].item()) for idx in pos_ids]
        neg_vals = [float(log_probs[idx].item()) for idx in neg_ids]
        logit_yes = logsumexp(pos_vals)
        logit_no = logsumexp(neg_vals)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "id": item_id,
            "primitive": "noul",
            "logit_yes": round(logit_yes, 6),
            "logit_no": round(logit_no, 6),
            "margin": round(logit_yes - logit_no, 6),
            "latency_ms": round(latency_ms, 2),
        }


def score_items_with_shared_prefix(
    model_path: str,
    model,
    tokenizer,
    prompt_style: str,
    default_prim: str,
    default_rubrics: str,
    default_choices: list,
    default_levels: list,
    default_debias: bool,
    pos_ids: list,
    neg_ids: list,
    items: list,
    stream_stdout: bool = False,
) -> list:
    from mlx_lm.models.cache import make_prompt_cache

    lm = model.language_model if hasattr(model, "language_model") else model
    common_context = items[0].get("context", "")
    prefix_prompt = f"<bos><start_of_turn>user\n【待评估背景材料 (Context)】\n{common_context}\n\n"
    prefix_ids = tokenizer.encode(prefix_prompt, add_special_tokens=False)

    base_cache = make_prompt_cache(lm)
    _ = lm(mx.array([prefix_ids]), cache=base_cache)
    mx.eval(*[s for c in base_cache for s in c.state if isinstance(s, mx.array)])

    results = []
    for item in items:
        t0_item = time.perf_counter()
        item_id = item.get("id", "")
        prim = item.get("primitive") or default_prim
        rubrics = item.get("rubrics") or default_rubrics
        debias = item.get("debias_order") if "debias_order" in item else default_debias

        if prim == "choice":
            choices = item.get("choices") or default_choices

            def run_choice_branch(ordered):
                lines = []
                for idx, c in enumerate(ordered):
                    tag = TAG_POOL[idx]
                    key = str(c.get("key", "")).strip()
                    desc = str(c.get("description", "")).strip()
                    lines.append(f"{tag}: {key} — {desc}" if desc and desc != key else f"{tag}: {key}")
                opts_block = "\n".join(lines)
                suffix = (
                    f"【分类指令 (Instructions)】\n{rubrics}\n\n"
                    f"【候选类别 (Criteria)】\n{opts_block}\n\n"
                    "【最佳类别代号（仅回答单个字符代号）】<end_of_turn>\n<start_of_turn>model\n"
                )
                suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                b_cache = [type(c).from_state(c.state) for c in base_cache]
                out = lm(mx.array([suffix_ids]), cache=b_cache)[0, -1, :]
                lp = out - mx.logsumexp(out)
                vals = {}
                for idx, c in enumerate(ordered):
                    tids = resolve_candidate_token_ids(tokenizer, [TAG_POOL[idx]])
                    vals[str(c.get("key", ""))] = logsumexp([float(lp[tid].item()) for tid in tids])
                sub_lse = logsumexp(list(vals.values()))
                return {k: v - sub_lse for k, v in vals.items()}

            pass1 = run_choice_branch(choices)
            if debias and len(choices) >= 2:
                pass2 = run_choice_branch(list(reversed(choices)))
                merged = {k: 0.5 * (pass1[k] + pass2[k]) for k in pass1}
                sub_lse = logsumexp(list(merged.values()))
                opt_logits = {k: round(v - sub_lse, 6) for k, v in merged.items()}
            else:
                opt_logits = {k: round(v, 6) for k, v in pass1.items()}

            res_obj = {
                "id": item_id,
                "primitive": "choice",
                "option_logits": opt_logits,
                "latency_ms": round((time.perf_counter() - t0_item) * 1000.0, 2),
            }
        else:
            suffix = (
                f"【判定准则 (Rubric)】\n{rubrics}\n\n"
                "【判定结果（仅回答“是”或“否”）】<end_of_turn>\n<start_of_turn>model\n"
            )
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            b_cache = [type(c).from_state(c.state) for c in base_cache]
            out = lm(mx.array([suffix_ids]), cache=b_cache)[0, -1, :]
            log_probs = out - mx.logsumexp(out)
            pos_vals = [float(log_probs[idx].item()) for idx in pos_ids]
            neg_vals = [float(log_probs[idx].item()) for idx in neg_ids]
            l_yes = logsumexp(pos_vals)
            l_no = logsumexp(neg_vals)
            res_obj = {
                "id": item_id,
                "primitive": "noul",
                "logit_yes": round(l_yes, 6),
                "logit_no": round(l_no, 6),
                "margin": round(l_yes - l_no, 6),
                "latency_ms": round((time.perf_counter() - t0_item) * 1000.0, 2),
            }

        results.append(res_obj)
        if stream_stdout:
            sys.stdout.write(json.dumps({"type": "item", "result": res_obj}, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    return results


def handle_score_request(req: dict, stream_stdout: bool = False) -> dict:
    model_path = req["model_path"]
    prompt_style = req.get("prompt_style", "gemma4-binary")
    default_prim = req.get("primitive", "noul")
    default_rubrics = req.get("rubrics", "")
    default_choices = req.get("choices", [])
    default_levels = req.get("score_levels", [])
    default_debias = bool(req.get("debias_order", False))
    default_image_tokens = int(req.get("image_tokens") or 280)
    items = req.get("items", [])

    if stream_stdout:
        sys.stdout.write(
            json.dumps({"type": "status", "status": "loading_model", "model_path": model_path}, ensure_ascii=False)
            + "\n"
        )
        sys.stdout.flush()

    model, tokenizer = MODEL_CACHE.get(model_path)
    if any(bool(it.get("images")) for it in items):
        MODEL_CACHE.get_vision_runner(model_path)
    pos_tokens = req.get("positive_tokens") or ["是", "Yes", "yes", "YES"]
    neg_tokens = req.get("negative_tokens") or ["否", "No", "no", "NO"]
    pos_ids = resolve_candidate_token_ids(tokenizer, pos_tokens)
    neg_ids = resolve_candidate_token_ids(tokenizer, neg_tokens)

    if stream_stdout:
        sys.stdout.write(
            json.dumps({"type": "status", "status": "scoring", "total": len(items)}, ensure_ascii=False) + "\n"
        )
        sys.stdout.flush()

    common_context = items[0].get("context", "") if items else ""
    is_shared_context = (
        len(items) >= 2
        and bool(common_context)
        and len(common_context) >= 60
        and not any(bool(it.get("images")) for it in items)
        and all(it.get("context") == common_context for it in items)
        and prompt_style in ("gemma4-binary",)
    )

    if is_shared_context:
        try:
            shared_results = score_items_with_shared_prefix(
                model_path,
                model,
                tokenizer,
                prompt_style,
                default_prim,
                default_rubrics,
                default_choices,
                default_levels,
                default_debias,
                pos_ids,
                neg_ids,
                items,
                stream_stdout=stream_stdout,
            )
            return {
                "ok": True,
                "type": "done",
                "mode": "shared_prefix_prefill",
                "primitive": default_prim,
                "model_path": model_path,
                "pos_token_ids": pos_ids,
                "neg_token_ids": neg_ids,
                "results": shared_results,
            }
        except Exception as exc:
            pass

    results = []
    for item in items:
        res_obj = score_single_item(
            model_path,
            model,
            tokenizer,
            prompt_style,
            default_prim,
            default_rubrics,
            default_choices,
            default_levels,
            default_debias,
            default_image_tokens,
            pos_ids,
            neg_ids,
            item,
        )
        results.append(res_obj)
        if stream_stdout:
            sys.stdout.write(json.dumps({"type": "item", "result": res_obj}, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    return {
        "ok": True,
        "type": "done",
        "primitive": default_prim,
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
