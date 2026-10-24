# 02 - macOS 本地推理基础设施与模型管理规范

## 1. 架构原则

1. **MLX 单步前向**：在 Apple Silicon 统一内存架构下，仅执行输入 Prompt 的单次前向（Prefill `max_tokens=1`），直接提取候选 Token 的 Logits，不进入自回归生成。
2. **系统资产零拷贝复用**：优先复用 `~/.cache/huggingface/hub/` 中的现有权重，避免重复下载与私有拷贝。
3. **常驻 Worker 进程**：Go CLI 通过 Unix Domain Socket (`~/.cache/decision-cli/worker.sock`) 与后台 Python MLX Worker 交互，模型常驻显存，空闲超时自动释放。

---

## 2. 预置模型配置

| 模型别名 | HuggingFace 仓库 | Prompt 风格 | 肯定候选 Token | 否定候选 Token | 显存占用 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma-4-26b-a4b` (推荐) | `mlx-community/gemma-4-26b-a4b-it-4bit` | `gemma4-binary` | `["是", "Yes", "yes"]` | `["否", "No", "no"]` | ~15.3GB |
| `ling-3.0-tiny` (轻量) | `rapid-mlx/Ling-3.0-tiny-MLX-4bit` | `ling-binary` | `["是", "Yes", "yes"]` | `["否", "No", "no"]` | ~4.8GB |
| `gemma-4-e4b` (轻量) | `mlx-community/gemma-4-e4b-it-4bit` | `gemma4-binary` | `["是", "Yes", "yes"]` | `["否", "No", "no"]` | ~3.5GB |
| `qwen3-reranker-4b` | `mlx-community/Qwen3-Reranker-4B-4bit` | `qwen3-reranker` | `["yes", "Yes"]` | `["no", "No"]` | ~2.5GB |

---

## 3. 运行时与路径解析契约

### 3.1 运行时解析优先级
1. `runtime.mlx_command`：用户显式指定的启动命令；
2. `runtime.python_bin`：用户显式指定的 Python 解释器；
3. `runtime.venv_dir`：独立虚拟环境（缺省 `~/.cache/decision-cli/venv`）；
4. 系统已有 Python 环境中已安装 `mlx_lm` 的解释器。

### 3.2 模型权重查找规则
1. **显式映射**：`config.yaml` 中 `models.paths[alias]`；
2. **扫描目录**：遍历 `models.scan_dirs`（包含 `~/.cache/huggingface/hub`、`~/models`），识别 HuggingFace snapshots 或平铺目录结构，定位含有权重与分词器的绝对路径。

---

## 4. Socket 通信契约

通过 Unix Domain Socket 传输 NDJSON：
- **请求**：
  ```json
  {"model_path": "...", "prompt_style": "...", "rubrics": "...", "items": [{"id": "...", "context": "..."}], "temperature": 1.0}
  ```
- **响应**：
  ```json
  {"ok": true, "results": [{"id": "...", "logit_yes": 4.12, "logit_no": 1.21, "probability": 0.948, "latency_ms": 18.2}]}
  ```

---

## 5. 环境自举流程 (`decision-cli setup`)

1. 检测 Apple Silicon (`arm64`) 与系统 Python / `uv`；
2. 在 `~/.cache/decision-cli/venv` 初始化独立虚拟环境并安装 `mlx-lm` 与 `huggingface_hub`；
3. 检查模型缓存，缺失时通过 `huggingface_hub` 下载至系统标准缓存目录。
