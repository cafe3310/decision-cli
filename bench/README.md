# decision-cli 基准评测套件 (bench)

`bench/` 目录是 `decision-cli` 的全量基准评测中心。它包含完整的 JevBench 子模块、已知评测数据档案与统一评测调度脚本 `run_bench`。

所有的中间评测数据集（如 Path A 中文 6 分类、5 档量规严重度评分、12 题中文二元测试）均由评测框架**现场动态生成**，无需依赖任何静态 `testdata` 文件。

---

## 1. 目录架构

```text
bench/
├── README.md                      # 本使用指南文档
├── run_bench                      # 统一评测调度入口脚本 (可直接执行)
├── jevbench/                      # Git 子模块 (TypeSafe AI 官方 JevBench 评测集与代码)
├── internal/                      # 评测内部执行套件
│   ├── dataset_generator.py       # 现场动态生成中间基准测试数据
│   ├── eval_language.py           # 语言评测套件 (Mini-bench + Path A + JevBench 231 题)
│   ├── eval_multimodal.py         # 多模态评测套件 (ImageJev-box 300 图像 × 3 赛道)
│   ├── compare_board.py           # JevBench 官方榜单快照对比工具
│   └── runner.py                  # 命令行参数解析与调度入口
└── results/                       # 评测结果档案库与摘要文档
    ├── summary.md                 # 已知模型表现与指标综合摘要文档
    ├── leaderboard/
    │   └── jevbench_v1.7.7_board.json   # 官方榜单快照 (295 个参与系统)
    ├── jevbench_public_231_eval_results.json
    ├── jevbench_multimodal_900_eval_results.json
    ├── path_a_multiclass_results.json
    ├── mini_bench_results.json
    └── local_vs_jev_benchmark_results.json
```

---

## 2. 统一评测脚本：`run_bench`

统一入口脚本 `bench/run_bench` 自动检测并优先使用项目自包含的 MLX 虚拟环境（`~/.cache/decision-cli/venv/bin/python3`），支持多套件、多模型、条数限制与榜单对比。

### 2.1 命令行参数速查

| 参数 | 缩写 | 默认值 | 可选值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `--suite` | `-s` | `all` | `language`, `multimodal`, `all` | 评测套件类型 |
| `--model` | `-m` | `gemma-4-26b-a4b` | `gemma-4-26b-a4b`, `gemma-4-e4b`, `qwen3-reranker-4b`, `all` | 目标评测模型，支持逗号分隔多个 |
| `--limit` | `-l` | `None` (全量) | 整数 (如 `2`, `5`, `10`) | 限制每个任务评测的题目条数（便于秒级快速回归） |
| `--compare-board` | - | `false` | 布尔开关 | 打印与 JevBench 官方排行榜的对比与排位 |
| `--output-dir` | `-o` | `bench/results`| 目录路径 | 评测生成的原始 JSON 结果保存目录 |
| `--cli-bin` | - | 自动探测 | 二进制路径 | 指定调用的 `decision-cli` 执行文件路径 |

---

## 3. 常见评测场景与命令示例

### 3.1 评测语言套件 (`--suite language`)
语言套件包含：中文二元判定 (Mini-bench)、Path A 中文互斥 6 分类（正逆向对冲）、5 档严重度打分以及 JevBench 231 题公开基准。

```bash
# 1. 快速验证推荐主力模型（每项任务抽查 2 题）：
./bench/run_bench --suite language --model gemma-4-26b-a4b --limit 2

# 2. 对特定轻量模型做全量语言评测：
./bench/run_bench --suite language --model gemma-4-e4b

# 3. 对全部支持的模型执行语言评测：
./bench/run_bench --suite language --model all
```

### 3.2 评测多模态套件 (`--suite multimodal`)
多模态套件基于 `bench/jevbench/datasets/imagejev-box-v0.1` 运行：
- **Track 1 (`noul`)**：视觉目标与 UI 元素验证（二元）；
- **Track 2 (`choice`)**：4 选 1 细粒度图像类别/UI 元素分类；
- **Track 3 (`choice`)**：4 选 1 空间候选框（Bounding Box）目标定位。

```bash
# 1. 快速验证多模态视觉链路（抽查 5 张图像）：
./bench/run_bench --suite multimodal --model gemma-4-26b-a4b --limit 5

# 2. 评测主力模型全量 300 张图像（900 题）：
./bench/run_bench --suite multimodal --model gemma-4-26b-a4b
```

### 3.3 全量端到端评测 (`--suite all`)

```bash
# 对推荐主力模型运行全量套件（语言 + 多模态）：
./bench/run_bench --suite all --model gemma-4-26b-a4b

# 对所有受支持模型运行全量套件：
./bench/run_bench --suite all --model all
```

### 3.4 与官方 JevBench 榜单对比 (`--compare-board`)

```bash
# 查阅本地已知评测指标与官方排行榜 295 款模型的横向对比：
./bench/run_bench --compare-board
```

---

## 4. 已知评测事实与结果档案

查看详细的历史评测事实、准确率对比与分层 ECE 误差，请参阅：
[`bench/results/summary.md`](file:///Users/sipan/workspace/decision-cli/bench/results/summary.md)。
