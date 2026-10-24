# decision-cli — Agent & Developer Usage Manual

Local Apple Silicon (MLX) non-autoregressive rubric decision CLI with deterministic SQLite content-addressable caching.

## 1. Quick Start (Single Item)

Evaluate whether a piece of text satisfies a natural-language rubric:

```bash
decision-cli \
  --rubrics "判断这些文本是否代表文档开发相关的需求" \
  --context "希望能提供一份完整的 API 接口参数说明和快速上手指南，现在的 README 太简略了"
```

### Standard JSON Output (`--format json`, default)

```json
{
  "decision": "是",
  "passed": true,
  "probability": 0.9999,
  "confidence": 0.9999,
  "threshold": 0.5,
  "temperature": 1.0,
  "logits": {
    "yes": -0.0001,
    "no": -14.25,
    "margin": 14.2499
  },
  "model": "gemma-4-26b-a4b",
  "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "cached": false,
  "latency_ms": 115.4,
  "cache_key": "a7f902c1..."
}
```

## 2. Batch Filtering & Exporting to YAML / JSON

Evaluate an entire `.yaml`, `.json`, or `.jsonl` dataset (including `ling-feedback` `cleaned-insights.yaml`):

```bash
decision-cli \
  --rubrics "判断这些文本是否代表文档开发相关的需求" \
  --input runs/2026-10-23/02-annotation-data/cleaned-insights.yaml \
  --output /tmp/doc-demands.yaml \
  --format yaml
```

Use `--filter-passed` to output only records where `decision == "是"`, or adjust `--threshold 0.75` to tighten precision:

```bash
decision-cli \
  --rubrics "判断这些文本是否包含商业采购或企业合作意向" \
  --input runs/2026-10-23/02-annotation-data/cleaned-insights.yaml \
  --threshold 0.75 \
  --filter-passed \
  --output /tmp/biz-leads.json
```

## 3. Global Configuration File (`~/.config/decision-cli/config.yaml`)

On first run, `decision-cli` automatically creates `~/.config/decision-cli/config.yaml` if it does not exist.
You can customize:

1. **MLX Command & Virtual Environment (`runtime`)**:
   - `runtime.mlx_command`: Custom command array (highest priority, e.g. `["conda", "run", "-n", "mlx", "python3"]`)
   - `runtime.python_bin`: Explicit Python executable with `mlx` and `mlx_lm` installed
   - `runtime.venv_dir`: Virtual environment directory (default `~/.cache/decision-cli/venv`, with automatic fallback to existing MLX venvs)
2. **Model Path Mapping, Scan Directories & Per-Model Calibration Temperatures (`models`)**:
   - `models.paths`: Map of `model-name -> /absolute/path/to/weights` (highest priority)
   - `models.scan_dirs`: Ordered list of directories scanned automatically for HuggingFace Hub snapshots (`models--<org>--<repo>/snapshots/<rev>`) or flat directories (defaults to `["~/.cache/huggingface/hub", "~/models", "~/.lmstudio/models"]`)
   - `models.registry.<model-name>.calibration_temperature`: Per-model empirical temperature scaling factor $T_{\text{noul}}$ used to calibrate $P(\text{yes}) = \sigma((\text{logit}_{\text{yes}} - \text{logit}_{\text{no}}) / T)$:
     - `gemma-4-26b-a4b`: `4.5` (post-RLHF MoE logits have average $|\text{margin}| \approx 16$; scaling by `4.5` prevents `0.9999` saturation while preserving the `P=0.5` decision boundary)
     - `gemma-4-e4b`: `3.5`
     - `qwen3-reranker-4b`: `1.0`
   - You can override `calibration_temperature` per model in `~/.config/decision-cli/config.yaml`, set global `default_temperature`, or pass `--temperature <float>` on the CLI.

## 4. Deterministic Cache Architecture (`~/.cache/decision-cli/cache.db`)

- **Cache Key Formula**:
  `SHA256("v1|" + canonical_model_id + "|" + normalize(rubrics) + "|" + normalize(context))`
- **Model Isolation**: Switching `--model` strictly uses a distinct `canonical_model_id` and never collides across models.
- **Parameter Decoupling**: The SQLite cache stores raw `logit_yes` and `logit_no`. Changing `--threshold` or `--temperature` hits the cache 100% and recalculates `probability` and `decision` in `< 1ms` without re-running the model.

## 5. Subcommands

- `decision-cli doctor`: Check Apple Silicon hardware, active `config.yaml`, MLX runtime, model path resolution, and run 3 canary inference tests with unambiguous expected outcomes.
- `decision-cli models`: List registered models and their local disk resolution status.
- `decision-cli cache stats`: Show SQLite cache statistics.
- `decision-cli cache export --output <file.yaml|file.json> [--rubrics <text>] [--model <name>]`: Export cached decisions to YAML or JSON.
- `decision-cli cache clear [--rubrics <text>] [--model <name>]`: Clear cached entries.
