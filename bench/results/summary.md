# decision-cli 基准评测结果与模型表现摘要

本文档归档与汇总 `decision-cli` 在各基准评测数据集上的已知模型评测事实。所有原始评测结果 JSON 均保存在本目录下。

---

## 1. 评测数据集与任务划分

| 评测套件 | 数据源 / 任务类型 | 样本量 | 核心度量指标 |
| :--- | :--- | :--- | :--- |
| **JevBench Public** | 官方公开基准 (`easy`, `standard`, `hard`) | 231 题 | 分层准确率、ECE 校准误差、Brier 分数、NRPS |
| **Multimodal 3-Tracks** | `imagejev-box-v0.1` (COCO, Doc, UI) | 900 题/模型 | Track 1 二元验证、Track 2 四选一分类、Track 3 空间框选 |
| **Path A 中文分类与打分** | 中文核心意图分类 + 5 档量规严重度打分 | 15 分类 × 2 顺序 + 10 打分 | 正逆序一致性（Order Invariance）、准确率、MAE |
| **Chinese Mini-Bench** | 中文二元意图判定（文档、量化部署、商业线索） | 12 题 | 二元准确率、Logit 概率校准 |

---

## 2. JevBench 官方公开 231 题评测事实 (`jevbench_public_231_eval_results.json`)

基于官方 JevBench 协议评估本地 3 款模型单步 Logit Prefill 性能：

| 模型名称 | 模型规模 | Easy Acc | Standard Acc | Hard Acc | 估计综合得分 (A) | 时延 p50 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma-4-26b-a4b`** (推荐主力) | 26B MoE (激活 3.8B) | **100.0%** | **100.0%** | **68.5%** | **65.23** | ~170ms |
| **`gemma-4-e4b`** (轻量备选) | 4.5B Dense | 97.9% | 97.2% | 42.1% | 23.26 | ~65ms |
| **`qwen3-reranker-4b`** (二元备选) | 4B Dense | 97.9% | 69.4% | 18.2% | 3.43 | ~52ms |

> [!NOTE]
> 在 JevBench 官方榜单快照（295 个参与系统）中，`gemma-4-26b-a4b` 的综合得分（65.23）位列开源与端侧模型顶级水准。

---

## 3. 多模态三赛道评测事实 (`jevbench_multimodal_900_eval_results.json`)

基于 `bench/jevbench/datasets/imagejev-box-v0.1` 中 300 张图像构建无数据泄露的 3 赛道评测（共 900 项判定/模型）：

| 评测赛道 | 任务性质 | `gemma-4-26b-a4b` 准确率 | `gemma-4-e4b` 准确率 |
| :--- | :--- | :--- | :--- |
| **Track 1: Visual Target Verification** | 二元判定 (`noul`, 50% 随机基线) | **92.3%** | 81.7% |
| **Track 2: Fine-Grained Classification** | 4 选 1 分类 (`choice`, 25% 随机基线) | **88.7%** | 74.3% |
| **Track 3: Spatial Grounding Box** | 空间候选框 4 选 1 (`choice`, 25% 随机基线) | **79.3%** | 58.7% |
| **三赛道总平均** | - | **86.8%** | 71.6% |

---

## 4. 中文业务多分类与分级打分事实 (`path_a_multiclass_results.json`)

针对中文业务意图分类（6 类）与 5 档严重度量规打分：

| 评估维度 | 指标定义 | `gemma-4-26b-a4b` | `gemma-4-e4b` | `qwen3-reranker-4b` |
| :--- | :--- | :--- | :--- | :--- |
| **正向顺序分类准确率** | Forward Choice Acc | **100.0%** (15/15) | 86.7% (13/15) | 53.3% (8/15) |
| **逆向顺序分类准确率** | Reverse Choice Acc | **100.0%** (15/15) | 80.0% (12/15) | 46.7% (7/15) |
| **正逆序一致性 (Invariance)** | Forward == Reverse | **100.0%** (15/15) | 86.7% (13/15) | 40.0% (6/15) |
| **分级量规精确准确率** | Score Exact Match | **90.0%** (9/10) | 70.0% (7/10) | 40.0% (4/10) |
| **分级打分平均绝对误差** | Score MAE | **0.10** | 0.30 | 1.10 |

---

## 5. 结论与选型指导

1. **主力通用与复杂判定首选**：`gemma-4-26b-a4b`。在中文逻辑、位置对冲一致性、分级量规单调性以及多模态图文与空间框选上全面领先；
2. **轻量与资源受限备选**：`gemma-4-e4b`。在 ~3.2GB 内存下提供 ~65ms 的极速推理，适合简单单步判别；
3. **二元检索重排专用**：`qwen3-reranker-4b`。仅适用于简单二元文本相关性场景，在多分类、量规打分与多模态场景不推荐使用。
