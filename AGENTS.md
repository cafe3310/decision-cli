# decision-cli 系统规范 (AGENTS.md)

## 1. 项目定位与目标

`decision-cli` 针对 macOS (Apple Silicon) 本地运行的决策评估任务设计。

系统利用 Apple MLX 统一内存架构，通过单次 Prefill 前向计算（`max_tokens=1`）读取目标判定 Token 的原始 Logits，经温度缩放或 Platt 仿射校准后输出确定性的二元或多元概率。系统配备本地 SQLite 内容寻址缓存，避免相同输入的重复前向计算。

---

## 2. 目录结构约定

```text
decision-cli/
├── README.md                                    # 面向使用者的快速上手、基准评测与命令指南
├── AGENTS.md                                    # 面向 AI Agent 的架构规范、模型调试与推理方案
├── TODO.md                                      # 任务进展跟踪
├── go.mod                                       # Go 模块定义
├── cmd/
│   └── decision-cli/main.go                     # CLI 唯一执行入口
├── internal/
│   ├── cli/                                     # 命令行解析、长参数契约、--usage 规范输出与 uninstall
│   ├── types/                                   # 数据模型、配置结构与接口协议定义
│   ├── infra/                                   # Python 虚拟环境管理、MLX 进程管理、模型解析与自检
│   ├── engine/                                  # Logits 提取、概率校准、去偏置与 Batch 调度
│   ├── cache/                                   # SQLite 内容寻址缓存与特征哈希
│   ├── config/                                  # 全局配置加载 (~/.config/decision-cli/config.yaml)
│   ├── server/                                  # HTTP API 服务 (/v1/systemone, /v1/noun)
│   ├── statusbar/                               # macOS 原生状态栏图标与内存状态监控
│   ├── tuner/                                   # 离线五折交叉验证与 Platt 仿射参数拟合
│   └── exporter/                                # 格式化输出与批量文件加载 (JSON/YAML/JSONL)
├── web/
│   └── tampermonkey/                            # 浏览器划选判定用户脚本与离线测试验证页
├── backlog/                                     # 历史需求、设计、评测与架构规范文档
│   └── specifications/                          # 长期生效的系统架构规范 (Spec 01~04)
├── bench/                                       # 评测套件源码、测试数据集与评测报表
└── tests/                                       # 端到端测试用例
```

---

## 3. 模型调试方案 (Debugging Framework)

当 Agent 遇到推理异常、未命中预期结果或守护进程状态异常时，按以下层级执行定位与调试：

### 3.1 守护进程与通信探活 (Socket-Level Probing)
1. **检查 Unix Domain Socket 状态**：
   - 默认套接字路径：`~/.cache/decision-cli/worker.sock`。
   - 探活机制：Go CLI 在发送推理请求前执行 `pingDaemon(sockPath)`（发送 `{"action":"ping"}`），超时时间为 500ms。
   - 若进程挂起或套接字无响应，CLI 会尝试获取 `~/.cache/decision-cli/worker.lock` 排他锁并重启 Worker。
2. **检查守护进程日志**：
   - 统一输出文件：`~/.cache/decision-cli/worker.log`。
   - 该日志记录 Python 进程启动、MLX 模块导入耗时、模型加载时间以及单步推理过程中的 Python Traceback。
3. **强制终止并清理残留进程**：
   ```bash
   decision-cli kill --force
   ```
   该命令通过系统进程表定位所有包含 `mlx_worker.py` 的进程并发送 `SIGKILL`，同时清理残留的 `.sock` 文件。

### 3.2 运行环境与基准样例诊断 (`decision-cli doctor`)
```bash
# 全局自检（检查硬件、配置、Python/MLX 版本与模型路径）
decision-cli doctor

# 指定模型自检（运行显著正向、负向与反转用例判定）
decision-cli doctor --model gemma-4-26b-a4b
```
- **判定标准**：
  - 正向用例（如明确的系统报错描述）输出 $P(\text{yes}) > 0.80$；
  - 负向用例（如无关闲聊）输出 $P(\text{yes}) < 0.20$；
  - 若基准测试得分未拉开差距，表明分词器 Token ID 映射错误、Prompt 模板错位或权重损坏。

### 3.3 缓存穿透排查 (`--no-cache` 与 `--cache-only`)
- 若怀疑结果来自陈旧缓存，使用 `--no-cache` 绕过读取并重写缓存：
  ```bash
  decision-cli noun --rubrics "..." --context "..." --no-cache
  ```
- 若需验证当前输入是否已计算过，使用 `--cache-only`（未命中直接返回退出码 1，不拉起模型）：
  ```bash
  decision-cli noun --rubrics "..." --context "..." --cache-only
  ```

---

## 4. 模型推理方案 (Inference Architecture)

### 4.1 部署机制与时延表现
在分类、审核与意图识别等判定任务中，目标通常是获取明确的分类标签与确定性概率。大语言模型在 Prefill 前向传播（输入 Prompt 处理）结束时，首个输出位置上已经包含了全词表各 Token 的未归一化对数几率（Logits）。

`decision-cli` 通过单次前向（`max_tokens=1`）直接提取目标候选 Token（如「是」与「否」或选项字母）对应的 Logits：
- 输出结构确定：直接对目标 Logits 计算概率，不进入自回归生成阶段。
- 推理时延低：前向计算只执行一次，单条判定在 Apple Silicon 上通常在 40ms 至 180ms 内完成。
- 内容寻址缓存：计算输入特征哈希并存入 SQLite，调整判定阈值或校准温度时无需再次调用模型，耗时小于 1ms。

### 4.2 单步 Prefill Logits 提取算法
针对二元判定（`noun`）：
1. 构造包含判定准则与待测上下文的结构化 Prompt，抑制模型自回归生成思维链（如 Gemma 4 采用 `<start_of_turn>user\n...\n<end_of_turn>\n<start_of_turn>model\n`）；
2. 执行单次 Prefill 前向传播，截取最后一位 Token 位置在全词表上的 Logits 向量：
   $$L = \text{Model.forward}(X)[-1, :]$$
3. 从词表中提取肯定候选集 $\mathcal{V}_{\text{pos}}$（如 `["是", "Yes", "yes"]`）与否定候选集 $\mathcal{V}_{\text{neg}}$（如 `["否", "No", "no"]`）的最大对数几率：
   $$l_{\text{yes}} = \max_{v \in \mathcal{V}_{\text{pos}}} L[v], \quad l_{\text{no}} = \max_{u \in \mathcal{V}_{\text{neg}}} L[u]$$
4. 计算边际差值：
   $$\Delta = l_{\text{yes}} - l_{\text{no}}$$

### 4.3 概率校准与 Platt 仿射校准
- **标准温度缩放**：
  $$P(\text{yes}) = \sigma\left(\frac{\Delta}{T}\right) = \frac{1}{1 + e^{-\Delta / T}}$$
- **Platt 仿射校准（解决基率偏移）**：
  $$z = A \cdot \Delta + B, \quad P(\text{yes}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
- **判定决策**：
  $$\text{Decision} = \begin{cases} \text{「是」 (true)}, & P(\text{yes}) \ge \tau \\ \text{「否」 (false)}, & P(\text{yes}) < \tau \end{cases}$$
  置信度定义为偏离随机无知（0.5）的确定程度：$\text{Confidence} = 2 \cdot |P(\text{yes}) - 0.5|$。

### 4.4 互斥多分类与正逆序去偏置 (`choose`)
1. 对 $K$ 个候选类别，分别提取对应选项索引 Token（如 `A, B, C...`）的 Logits：$\{l_1, l_2, \dots, l_K\}$。
2. 归一化 Softmax 概率：
   $$P_k = \frac{e^{l_k / T}}{\sum_{j=1}^K e^{l_j / T}}$$
3. **位置去偏置 (`--debias-order`)**：
   大语言模型存在首位或末位选项偏置。开启 `--debias-order` 时，系统执行两次前向计算：正序排列与倒序排列。将两次推理得到的对应类别未归一化 Logits 取算术均值后再执行 Softmax：
   $$l_k^{\text{debiased}} = \frac{l_k^{\text{forward}} + l_k^{\text{reverse}}}{2}$$

### 4.5 共享前缀 KV Cache 预填充 (Shared Prefill)
在 `batch` 或 `systemone` 复合多问场景中，多个判断共享相同的一级准则（`rubrics`）或较长背景文本（`context`）。
- MLX Worker 利用 `make_prompt_cache` 预先对共享前缀进行 Prefill 并缓存 KV 状态；
- 对每个子问题分支仅需对差异后缀进行增量前向计算，计算复杂度由 $\mathcal{O}(N \times (L_{\text{shared}} + L_{\text{query}}))$ 降至 $\mathcal{O}(L_{\text{shared}} + N \times L_{\text{query}})$。

### 4.6 Agent 调用准则：批量文件优先 (Batch-First Rule)
- 避免在 Shell 脚本或代码循环中逐条反复拉起 `decision-cli noun`。多次子进程创建与参数解析会带来不必要的开销；
- 当处理样本数量 $\ge 2$ 时，将数据写入临时 `.json` 或 `.jsonl` 文件，调用 `decision-cli batch noun --input <file>` 一并提交；
- 批量模式直接在常驻 Worker 内复用显存缓冲区与连接，吞吐速度显著优于单条独立执行。

---

## 5. 退出码与错误契约

| 退出码 | 含义 | 说明 |
| :---: | :--- | :--- |
| `0` | 判定成功 / 命令正常执行 | 标准输出中输出请求的结构化数据（JSON/YAML/表格） |
| `1` | 业务不匹配 / 缓存未命中 | 在 `--cache-only` 模式下未命中缓存，或二元判定返回“否”且指定了条件退出 |
| `2` | 命令行参数或输入格式错误 | 缺少必需的 `--rubrics`、`--context` 或 `--input` 文件不存在 |
| `3` | 运行环境或依赖异常 | Python 虚拟环境损坏、MLX 导入失败或硬件不支持 |
| `4` | 模型路径未解析或权重缺失 | 指定的模型未在扫描目录或配置中找到 |
