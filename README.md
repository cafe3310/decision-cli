# decision-cli

本地部署决策模型，通过命令行提供批量处理与 Jev-like API，基于 MLX 运行在 Apple Silicon 上，支持文本与多模态。

---

## 1. 原理简述

在分类、审核与意图识别等判定任务中，目标通常是获取明确的分类标签与确定性概率。

模型在处理输入 Prompt 的首步前向计算（Prefill 阶段，`max_tokens=1`）后，输出层已给出词表所有 Token 的未归一化对数几率（Logits）。

`decision-cli` 在单步前向中直接提取候选 Token（如「是」与「否」或选项字母）的 Logits 计算相对概率。单条计算仅需单次前向，耗时约数十至百余毫秒（端侧轻量模型约 45ms）。相同输入或调整判定阈值时读取本地 SQLite 缓存，耗时小于 1ms。

---

## 2. 评测结果

本测试用于评估「将 LLM 用于 Decision Model」的非微调部署表现。LLM 用于 Decision 任务的上限远不止于此，若开展针对性微调可以进一步优化效果。

### 2.1 JevBench 公开题测试

基于标准 JevBench 评测集的公开题目，评估各模型在单步 Logit Prefill 模式下的表现：

| 模型名称 | 模型类型 / 激活参数量 | Easy Acc (48题) | Standard Acc (72题) | Hard Acc (111题) | 综合准确率 (231题) | 单条耗时 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ling-3.0-tiny (轻量) | MoE+线性注意力 (总 7.9B / 激活 1.3B) | 97.9% | 83.3% | 48.6% | 69.7% | ~45 ms |
| qwen3-reranker-4b | Dense (4B) | 97.9% | 69.4% | 35.1% | 58.9% | ~52 ms |
| gemma-4-e4b (轻量) | Dense (4.5B) | 97.9% | 97.2% | 50.5% | 74.9% | ~65 ms |
| gemma-4-26b-a4b (主力) | MoE (总 26B / 激活 3.8B) | 100.0% | 100.0% | 68.5% | 84.8% | ~170 ms |

> [!NOTE]
> 评测数据基于官方公开 JevBench 231 题（Easy 48 题 / Standard 72 题 / Hard 111 题）。各模型单步前向原始结果归档于 `bench/results/jevbench_public_231_eval_results.json`（`ling-3.0-tiny` 归档于 `bench/results/bench_run_20261030_201008.json`）。全量复现评测可执行 `python3 bench/run_full_eval.py`。

### 2.2 多模态视觉评测 (900 项判定)

基于标准图像评测集评测视觉多模态判定准确率：

| 评测赛道 | 任务类型 | gemma-4-26b-a4b | gemma-4-e4b |
| :--- | :--- | :--- | :--- |
| Track 1: 视觉目标验证 | 二元判定 (`noun`, 随机基线 50%) | 92.3% | 81.7% |
| Track 2: 细粒度分类 | 4 选 1 分类 (`choose`, 随机基线 25%) | 88.7% | 74.3% |
| Track 3: 空间候选框选择 | 空间候选框 4 选 1 (`choose`, 随机基线 25%) | 79.3% | 58.7% |
| 三赛道均值 | - | 86.8% | 71.6% |

---

## 3. 安装与环境准备

运行需要 Apple Silicon Mac，macOS 13.0 以上，已安装 Xcode 命令行工具、Go 1.22+ 以及 uv（若未安装 uv 会自动回退使用 `venv`）。

### 3.1 编译与安装二进制
```bash
git clone https://github.com/cafe3310/decision-cli.git
cd decision-cli
go build -o bin/decision-cli ./cmd/decision-cli
export PATH="$(pwd)/bin:$PATH"
```

### 3.2 初始化自包含运行时环境
执行 `setup` 命令在 `~/.cache/decision-cli/venv` 创建独立虚拟环境，并安装 `mlx`、`mlx-lm`、`pillow` 与 `huggingface_hub`（不影响系统全局 Python 环境）：
```bash
decision-cli setup
```

### 3.3 环境与基准样例自检
执行 `doctor` 验证硬件信息、环境依赖及单步推理逻辑正确性：
```bash
decision-cli doctor
```

---

## 4. 模型管理与下载

`decision-cli` 自动扫描本地 Hugging Face 缓存目录（`~/.cache/huggingface/hub/`）与常见模型路径。若本地未就绪，可通过内置命令下载：

```bash
# 查看支持的模型及其本地路径解析状态
decision-cli models

# 下载轻量端侧模型（约 4.8GB，纯文本高吞吐）
decision-cli models download ling-3.0-tiny

# 下载推荐主力模型（约 14.8GB，支持文本与视觉多模态）
decision-cli models download gemma-4-26b-a4b

# 下载轻量备选模型（约 3.2GB，支持多模态）
decision-cli models download gemma-4-e4b
```

---

## 5. 主要用法

### 5.1 二元判定 (`noun`)
```bash
# 纯文本判定
decision-cli noun \
  --rubrics "判断用户反馈是否属于支付流程受阻的严重报错" \
  --context "微信支付点击后一直转圈，最后提示错误码 5002，已扣款但未生成订单"

# 图文多模态判定
decision-cli noun \
  --rubrics "判断截图中是否显示 HTTP 502 错误弹窗" \
  --image ./screenshot.png
```

### 5.2 多分类 (`choose`)
```bash
# 多分类判定（支持 --debias-order 消除选项位置偏置）
decision-cli choose \
  --rubrics "判断用户反馈的诉求类型" \
  --choices "bug:系统故障报错,feature:新功能建议,sales:商务咨询与报价" \
  --context "请问系统支持离线私有化部署吗？费用如何计算？" \
  --debias-order
```

### 5.3 分级打分 (`score`)
```bash
decision-cli score \
  --rubrics "评估用户情绪负面程度" \
  --levels "平静礼貌,轻微催促,语气急躁,严重愤怒使用攻击性语言" \
  --context "我都等了半小时了，每次都是机器人自动回复，到底有没有人处理？"
```

### 5.4 批量处理 (`batch`)
对数据文件（`.json`, `.jsonl`, `.yaml`）执行批量流式判定，避免逐条调用的进程启动开销：
```bash
decision-cli batch noun \
  --rubrics "判断内容是否包含低俗或恶意推广信息" \
  --input ./data.jsonl \
  --output ./filtered.jsonl \
  --filter-passed \
  --progress
```

### 5.5 常驻 HTTP 服务 (`serve`)
启动兼容 TypeSafe Jev `/v1/systemone` 及 `/v1/noun` 端点的 HTTP 服务：
```bash
decision-cli serve --host 127.0.0.1 --port 8080
```

---

## 6. 清理与卸载

执行 `uninstall` 命令停止后台常驻进程，并清除本机生成的所有缓存、虚拟环境与临时文件：

```bash
# 终止后台 Worker 与状态栏，清理 ~/.cache/decision-cli 虚拟环境与 SQLite 缓存
decision-cli uninstall

# 非交互式执行，并一并移除全局配置文件 ~/.config/decision-cli
decision-cli uninstall --yes --all
```

若需移除编译产物，可执行：
```bash
trash $(which decision-cli)
```

---

## 7. 目录结构

```text
decision-cli/
├── README.md                      # 项目说明文档与基准评测报告
├── AGENTS.md                      # 面向 AI Agent 的开发调试与推理契约
├── TODO.md                        # 任务进展跟踪
├── cmd/decision-cli/main.go       # 命令行入口
├── internal/
│   ├── cli/                       # 命令行参数解析、长参数契约与使用说明
│   ├── engine/                    # Logits 提取、概率校准与调度引擎
│   ├── server/                    # HTTP API 服务 (/v1/systemone, /v1/noun)
│   ├── infra/                     # Python 虚拟环境管理、MLX 进程控制与模型解析
│   ├── cache/                     # SQLite 内容寻址缓存
│   ├── config/                    # 全局配置管理 (~/.config/decision-cli/config.yaml)
│   ├── statusbar/                 # macOS 状态栏图标与内存状态监控
│   ├── tuner/                     # 离线五折交叉验证与 Platt 仿射校准
│   └── types/                     # 数据结构与类型定义
├── web/
│   └── tampermonkey/              # 浏览器划选判定用户脚本与离线测试页
├── backlog/                       # 历史需求、设计与架构规范
│   └── specifications/            # 长期架构规范 (Spec 01~04)
├── bench/                         # 评测套件、结果数据与可视化报表
└── tests/                         # 端到端测试用例
```

---

## 8. 方案来源与公开引用

单步概率提取、校准与缓存方案均基于公开文献与开源实现。

感谢 [@Appointat](https://github.com/Appointat) 分享的概率分布优化方案。

### 8.1 决策模型与单步 Logit 判别
- TypeSafe AI Jev 与 JevBench：通过单步 Prefill 前向读取 Token Logits 的决策评估协议与公开基准 ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench))。
- kev：System-1 决策 API 开源实现 ([jaredpalmer/kev](https://github.com/jaredpalmer/kev))。
- Pointwise Logit 判别与 Reranker：在 Prompt 末尾提取候选 Token 未归一化对数几率进行判别的工程实践，如 [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-4B-4bit)、[Qwen3-VL-Reranker](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B)、[Jina-Reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0)。

### 8.2 概率校准
- 温度缩放：通过单一温度标量调整概率平滑度与置信度。参考 Chuan Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599))。
- Platt 仿射校准：通过逻辑回归参数 $(A, B)$ 修正先验基率偏移。参考 John C. Platt, *Probabilistic Outputs for Support Vector Machines*, 1999。

### 8.3 选项排列偏置对冲
- 大模型评估时存在选项位置偏好，通过交换候选选项顺序并对未归一化 Logits 取均值消除偏置。参考 Peiyi Wang et al., *Large Language Models are not Fair Evaluators*, ACL 2023 ([arXiv:2305.17926](https://arxiv.org/abs/2305.17926))。

### 8.4 共享前缀 KV Cache 预填充
- 对多题判定复用共同准则或背景上下文的注意力状态。参考 In Gim et al., *Prompt Cache: Modular Attention Reuse for Low-Latency LLM Inference*, MLSys 2024 ([arXiv:2311.04934](https://arxiv.org/abs/2311.04934))。

### 8.5 底层框架与开源模型
- Apple MLX：针对 Apple Silicon 统一内存架构的推理框架 ([ml-explore/mlx](https://github.com/ml-explore/mlx))。参考 Awni Hannun et al., *MLX: Efficient Machine Learning on Apple Silicon*, 2023。
- 开源基础模型：Google Gemma、InclusionAI Ling-3.0、Alibaba Cloud Qwen。

---

## 9. 致谢

感谢 [@Appointat](https://github.com/Appointat) 分享的优化方案。

---

## 10. 开源协议

[Apache-2.0](LICENSE)
