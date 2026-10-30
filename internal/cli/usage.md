# decision-cli — AI Agent 与开发者中文使用手册

`decision-cli` 是一款专为 macOS (Apple Silicon) 打造的本地高性能非自回归决策模型（System-1 Decision Model）命令行工具。它基于 Apple MLX 框架，通过单步前向推理（Prefill Logits，`max_tokens=1`）直接提取目标 Token 的原始 Logits 并计算校准概率，配合本地 SQLite 内容寻址缓存，实现毫秒级判定响应。

---

## 模型选型与多模态支持（默认主力：`gemma-4-26b-a4b`）

| 模型别名 | 底层权重 | 模态能力 | 内存占用 | 定位与推荐场景 |
|---|---|---|---|---|
| **`gemma-4-26b-a4b`** (默认主力) | `mlx-community/gemma-4-26b-a4b-it-4bit` | **多模态 (文本 + 单图/多图)** | ~15.3GB | 26B MoE（每 Token 激活 3.8B）+ 27 层 1152 维 Vision Tower。逻辑与视觉判别精度最高，优先推荐 |
| **`ling-3.0-tiny`** (端侧轻量) | `rapid-mlx/Ling-3.0-tiny-MLX-4bit` | **纯文本 (不支持 `--image`)** | ~4.8GB | 7.9B MoE（激活 1.3B）+ 3:1 KDA 线性注意力。前向速度最快（~45ms），适合纯文本高吞吐场景 |
| **`gemma-4-e4b`** (轻量备选) | `mlx-community/gemma-4-e4b-it-4bit` | **多模态 (文本 + 单图/多图)** | ~3.5GB | 4.5B Dense + 16 层 768 维 Vision Tower。适用于受限内存环境下的文本与视觉轻量判别 |
| **`qwen3-reranker-4b`** (纯文本备选) | `mlx-community/Qwen3-Reranker-4B-4bit` | **纯文本 (不支持 `--image`)** | ~2.5GB | 交叉编码器模型，仅适合纯文本简单二元检索匹配（传入 `--image` 会报错拦截） |

- **自包含 Python/MLX 环境 (`~/.cache/decision-cli/venv`)**：
  - 项目默认使用完全自包含的 `~/.cache/decision-cli/venv` 虚拟环境（内含 `mlx`、`mlx-lm`、`pillow`），不依赖外部工作区环境。首次运行自动创建，也可随时运行 `decision-cli setup` 校验或修复。

---

## Agent 调用准则：批量文件优先 (Batch-First Rule)

AI Agent 在需要执行多次或批量判断时，**严禁使用 Shell 循环逐条调用单条子命令**（如 `for x in ...; do decision-cli noun ...; done`）。

### 为什么必须使用批量文件模式：
1. **吞吐提升 5~10 倍**：批量模式在底层复用 KV Cache，采用连续流式单步推理，避免反复建立进程和 Socket 握手；
2. **全局队列与状态栏监控**：批量提交的文件会被全局唯一 MLX Worker 纳入队列，macOS 菜单栏状态图标可实时观测整体进度、剩余队列与吞吐速度；
3. **自动批量缓存寻址**：CLI 会对批量输入进行多并发 SQLite 缓存比对，已命中项毫秒级直接返回，未命中项自动聚合成紧凑批次送入 GPU。

### 标准批量调用模版：
1. 将待判定的多条数据组织为本地临时文件（推荐 `.jsonl` 或 `.json`）：
   ```json
   {"id": "item-001", "context": "第一段待评估文本..."}
   {"id": "item-002", "context": "第二段待评估文本..."}
   ```
2. 调用对应的 `batch` 子命令处理，配合 `--progress` 输出实时流式进度：
   ```bash
   decision-cli batch noun \
     --rubrics "判断文本是否为系统报错" \
     --input ./items.jsonl \
     --output ./results.jsonl \
     --format jsonl \
     --progress
   ```

---

## ⚠️ 选项分隔符与特殊字符注意事项

在命令行传递 `--choices` 与 `--levels` 时：
1. **`--choices` 采用英文半角逗号 `,` 分隔选项，冒号 `:` 分隔 `键:描述`**：
   - 格式规范：`--choices "key1:描述一,key2:描述二,key3:描述三"`
   - 英文逗号 `,` 分隔不同候选类别；
   - 英文冒号 `:` 分隔选项标识符（key）与具体含义描述。
   - **Agent 注意事项**：选项键或描述内部如需标点，请使用全角中文标点（`，`、`：`、`；`）或斜杠 `/`，避免被切分破坏。
2. **`--levels` 采用英文半角逗号 `,` 分隔递增等级刻度**：
   - 格式规范：`--levels "等级0描述,等级1描述,等级2描述"`（支持 2 到 10 个升序分级，对应 0 到 $M-1$）。
   - **Agent 注意事项**：等级描述内部不要包含英文半角逗号 `,`，可用中文逗号 `，` 替代。

---

## 一、单条文本判别子命令

单条命令仅接受字符串输入，不接受文件路径。

### 1. `decision-cli noun` — 二元判定（是/否）

判定给定文本是否满足自然语言准则：

```bash
decision-cli noun \
  --rubrics "判断该反馈是否涉及支付流程的严重系统报错" \
  --context "微信支付点击确认后一直转圈，最后提示错误码 5002，钱扣了但订单未生效"
```

**JSON 响应示例**：
```json
{
  "decision": "是",
  "passed": true,
  "probability": 0.9842,
  "confidence": 0.9842,
  "threshold": 0.5,
  "temperature": 4.5,
  "logits": {
    "yes": 4.125,
    "no": -6.218,
    "margin": 10.343
  },
  "model": "gemma-4-26b-a4b",
  "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "cached": false,
  "latency_ms": 168.5,
  "cache_key": "c3a8b2..."
}
```

- 若仅需要标签，可指定 `--format label-only`，输出纯文本 `是` 或 `否`。
- 若需要紧凑输出，可指定 `--format compact`，输出 `[gemma-4-26b-a4b] 是 (p=0.9842, conf=0.9842, 168ms)`。
- **多模态单图/多图支持（`--image` & `--image-tokens`）**：
  - 支持传入单张或多张本地图片路径或 `data:image/...;base64,...`：
    ```bash
    # 单图视觉二元判定（传入 --image 时 --context 可省略）：
    decision-cli noun --rubrics "判断截图中是否包含报错弹窗" --image ./error.png

    # 多图对比判定（使用 {{image:1}}、{{image:2}} 显式指定图片位置）：
    decision-cli noun \
      --rubrics "对比 {{image:1}}（修复前）与 {{image:2}}（修复后），判断文字重叠 Bug 是否已解决" \
      --image ./before.png --image ./after.png \
      --image-tokens 280
    ```
  - `--image-tokens` 控制单张图片的视觉 Soft Token 预算，可选 `70 | 140 | 280 | 560 | 1120`（默认 `280`；高密度小字 OCR 推荐 `560` 或 `1120`，多图快速对比推荐 `70` 或 `140`）。

---

### 2. `decision-cli choose` — 互斥多分类选择

在多个给定候选类别中，计算归一化 Softmax 概率并选出最符合的类别：

```bash
decision-cli choose \
  --rubrics "判断该用户反馈所属的诉求类型" \
  --choices "bug:系统故障或白屏报错,feature:新功能提议或体验优化,sales:商业咨询或私有化部署,noise:闲聊无明确诉求" \
  --context "请问系统支持离线私有化部署到医疗内网吗？大概费用怎么核算？" \
  --debias-order
```

**JSON 响应示例**：
```json
{
  "primitive": "choice",
  "choice": "sales",
  "confidence": 0.9412,
  "probabilities": {
    "bug": 0.0081,
    "feature": 0.0295,
    "noise": 0.007,
    "sales": 0.9412
  },
  "logits": {
    "bug": -4.75,
    "feature": -1.52,
    "noise": -5.12,
    "sales": -0.02
  },
  "temperature": 2.5,
  "model": "gemma-4-26b-a4b",
  "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "cached": false,
  "latency_ms": 172.1,
  "cache_key": "8e41bf..."
}
```

- `--debias-order`：运行正序加逆序双向推理并对冲位置偏置，推荐在重要分类任务与视觉空间定位任务中开启。
- **多模态视觉分类、双图对比与候选 Box / 九宫格空间定位（`choose --image`）**：
  - **场景 A：图像空间定位 / 九宫格候选 Box 选择（重点推荐范式）**
    - 由于 `decision-cli` 是 `max_tokens=1` 的非自回归单步概率决策模型，当需要定位图像中某个目标的位置或评估九宫格各区域（如“哪个区域最帅”、“目标位于哪个候选框”）时，**标准做法是将候选区域或候选边界框（Bounding Box）作为 `--choices` 选项传入，让模型单步计算每个候选 Box 的概率**。
    - **Gemma 4 原生坐标约定**：采用 `0..1000` 归一化整数坐标 `[ymin，xmin，ymax，xmax]`（以图像左上角为原点 `[0，0]`，右下角为 `[1000，1000]`）。
    - **⚠️ 分隔符防踩坑**：`--choices` 使用英文半角逗号 `,` 切分不同选项，因此**坐标方括号内部必须使用中文全角逗号 `，`**（或者改用 `decision-cli systemone --input req.json` 传递 JSON）：
      ```bash
      decision-cli choose \
        --model gemma-4-26b-a4b \
        --image "/path/to/photo.jpeg" \
        --image-tokens 560 \
        --debias-order \
        --rubrics "将这张图按 3行×3列 平均分成 9 个方块区域（区域1至区域9）。在这 9 个方块区域中，哪个区域最帅？" \
        --choices "grid_1:区域1（左上角 [0，0，333，333]）,grid_2:区域2（中上 [0，333，333，667]）,grid_3:区域3（右上角 [0，667，333，1000]）,grid_4:区域4（左中 [333，0，667，333]）,grid_5:区域5（正中央 [333，333，667，667]）,grid_6:区域6（右中 [333，667，667，1000]）,grid_7:区域7（左下角 [667，0，1000，333]）,grid_8:区域8（中下 [667，333，1000，667]）,grid_9:区域9（右下角 [667，667，1000，1000]）" \
        --context "请观察图像中人物五官、发型与穿搭所在的九宫格位置，选出最帅的那个方块区域。"
      ```
  - **场景 B：双图 / 多图 A/B 视觉方案对比**
    ```bash
    decision-cli choose \
      --rubrics "对比两张 UI 截图的排版留白与信息层级可读性，选出更优版本" \
      --choices "plan_a:第一张图更清晰易读,plan_b:第二张图更清晰易读" \
      --image ./ui_v1.png --image ./ui_v2.png \
      --context "方案 A：{{image:1}}\n方案 B：{{image:2}}" \
      --debias-order
    ```

---

### 3. `decision-cli score` — 分级量规连续打分

依据多级阶梯描述，评估文本处于哪一个等级，并输出数学期望均值与等级离散概率分布：

```bash
decision-cli score \
  --rubrics "评估用户客服对话中表达出的负面情绪严重程度" \
  --levels "完全平静礼貌,轻微催促质疑,语气急躁指责,严重愤怒使用攻击性语言" \
  --context "你们客服到底看没看我发的消息？我都等了半小时了，每次都是机器人自动回复敷衍！" \
  --debias-order
```

**JSON 响应示例**：
```json
{
  "primitive": "score",
  "score": 2,
  "expected_score": 1.942,
  "confidence": 0.8125,
  "level_probabilities": [
    0.0012,
    0.1241,
    0.8125,
    0.0622
  ],
  "levels": [
    "完全平静礼貌",
    "轻微催促质疑",
    "语气急躁指责",
    "严重愤怒使用攻击性语言"
  ],
  "temperature": 1.5,
  "model": "gemma-4-26b-a4b",
  "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "cached": false,
  "latency_ms": 170.8,
  "cache_key": "9d27aa..."
}
```

---

## 二、批量文件处理子命令 (`batch`)

批量命令统一由 `decision-cli batch` 引导，**仅接受数据文件路径输入**（通过 `--input`），严禁传入 `--context`。支持 `.json`、`.jsonl`、`.yaml`、`.yml` 格式。

### 批量通用参数清单

| 参数 | 类型 | 默认值 | 作用说明 |
|---|---|---|---|
| `--input` | 字符串 | **必填** | 输入数据文件路径（支持 JSON 数组、JSONL、YAML 列表） |
| `--output` | 字符串 | `""` | 结果输出保存路径（缺省打印到 stdout） |
| `--format` | 字符串 | `"json"` | 输出格式：`json` \| `yaml` \| `jsonl` |
| `--id-field` | 字符串 | `"id"` | 每条记录中的唯一主键字段名称 |
| `--context-template`| 字符串 | `""` | Go 模板字符串，自定义提取多字段拼装 context |
| `--image-field` | 字符串 | `""` | 记录中图片路径或图片数组的字段名（缺省自动识别 `images` / `image`） |
| `--image-base-dir` | 字符串 | `""` | 解析记录中相对图片路径的基准目录（缺省相对于输入文件所在目录） |
| `--image-tokens` | 整数 | `280` | 单张图片的视觉 Soft Token 预算（`70 \| 140 \| 280 \| 560 \| 1120`） |
| `--include-record` | 布尔值 | `false` | 输出中是否附带输入的原始记录对象 |
| `--progress` | 布尔值 | `false` | 是否在 stderr 流式打印逐条处理进度（JSONL） |
| `--model` | 字符串 | `"gemma-4-26b-a4b"` | 模型别名或物理路径 |
| `--temperature` | 浮点数 | 按配置 | 校准温度覆盖值 |
| `--no-cache` | 布尔值 | `false` | 绕过 SQLite 缓存读取与写入 |
| `--cache-only` | 布尔值 | `false` | 仅查本地缓存，不调用 MLX 后端推理 |

---

### 1. `decision-cli batch noun` — 批量二元判定

支持使用 `--filter-passed` 仅输出通过（判定为「是」）的条目：

```bash
decision-cli batch noun \
  --rubrics "判断该反馈是否涉及数据安全或隐私泄露风险" \
  --input /path/to/feedbacks.jsonl \
  --output /path/to/security-alerts.json \
  --filter-passed \
  --include-record \
  --progress
```

**批量输出结果结构 (`security-alerts.json`)**：
```json
{
  "rubrics": "判断该反馈是否涉及数据安全或隐私泄露风险",
  "model": "gemma-4-26b-a4b",
  "threshold": 0.5,
  "summary": {
    "total": 500,
    "passed_count": 12,
    "rejected_count": 488,
    "cache_hits": 450,
    "cache_misses": 50,
    "total_time_ms": 8420.5
  },
  "results": [
    {
      "id": "item-042",
      "decision": "是",
      "passed": true,
      "probability": 0.9654,
      "confidence": 0.9654,
      "cached": true,
      "record": {
        "id": "item-042",
        "user_expression": "员工导出明细时竟然能直接下载未经脱敏的客户身份证号和手机号！",
        "user_name": "张经理"
      }
    }
  ]
}
```

---

### 2. `decision-cli batch choose` — 批量多分类标注

```bash
decision-cli batch choose \
  --rubrics "用户反馈问题归因" \
  --choices "ui:界面设计缺陷,backend:服务端接口异常,network:本地网络超时,doc:文档说明不全" \
  --input tickets.json \
  --output classified-tickets.json \
  --debias-order
```

---

### 3. `decision-cli batch score` — 批量分级打分

```bash
decision-cli batch score \
  --rubrics "工单处理紧急度评级" \
  --levels "P3常规低优,P2重要跟进,P1高优阻断,P0系统崩溃即刻响应" \
  --input bug-reports.yaml \
  --output bug-scores.yaml \
  --format yaml \
  --debias-order
```

---

## 三、复合问卷评估 (`decision-cli systemone`)

遵循 TypeSafe Jev `/v1/systemone` 接口契约。在单份共享上下文（`state`）上，一次性前向评估多个不同类型的混合判定问题（支持二元 `noul`、选择 `choice`、分级 `score` 并存）：

```bash
decision-cli systemone \
  --input request.json \
  --output response.json \
  --debias-order
```

**输入请求 (`request.json`) 示例**：
```json
{
  "state": "用户在结算页点击微信支付后返回错误码 5002，微信零钱已扣减但系统显示未支付，用户连续提交了 3 次工单并语气急促催促退款。",
  "questions": [
    {
      "id": "q_financial_loss",
      "primitive": "noul",
      "instruction": "该问题是否导致了用户的实际资金损失？"
    },
    {
      "id": "q_urgency_level",
      "primitive": "score",
      "instruction": "评估该工单的紧急处理级别",
      "levels": ["P3日常", "P2中度", "P1紧急", "P0特急"]
    },
    {
      "id": "q_issue_domain",
      "primitive": "choice",
      "instruction": "归纳该问题所属业务域",
      "choices": [
        {"key": "payment", "description": "支付与财务系统"},
        {"key": "order", "description": "订单与履约中心"},
        {"key": "account", "description": "账号与登录中心"}
      ]
    }
  ]
}
```

---

## 四、系统自检与运维子命令

### 1. `decision-cli doctor` — 环境健康自检与基准样例推理

自动检查 Apple Silicon 硬件、统一内存、配置文件、Python 虚拟环境与 MLX 版本，并对受支持模型执行显著用例（正向例、负向例、反转例）的基准样例单步推理诊断：

```bash
decision-cli doctor
# 仅对指定模型进行测试：
decision-cli doctor --model gemma-4-26b-a4b
```

### 2. `decision-cli models` — 模型探测、路径解析与权重下载

展示支持的模型配置、本地磁盘缓存状态（`READY` / `MISSING`）及解析来源：

```bash
decision-cli models
# 输出 JSON 格式：
decision-cli models --format json

# 从零下载模型权重至本地 scan_dirs 扫描目录：
decision-cli models download gemma-4-26b-a4b
decision-cli models download gemma-4-e4b
```

### 3. `decision-cli cache` — SQLite 缓存管理

- **查看统计**：`decision-cli cache stats --format json`
- **导出缓存**：`decision-cli cache export --rubrics "数据安全" --format yaml --output cached.yaml`
- **清空缓存**：`decision-cli cache clear`（或指定 `--rubrics` / `--model` 定向清除）。

### 4. `decision-cli serve` — 启动常驻 SystemOne HTTP API 服务

拉起轻量 HTTP Web 服务，保持监听指定端口，暴露与 TypeSafe Jev 协议兼容的标准接口以及快捷单条接口：

```bash
# 在指定 Host:Port 启动服务：
decision-cli serve --host 127.0.0.1 --port 8080 --model gemma-4-26b-a4b
```

提供的接口：
- `POST /v1/systemone`：标准复合多问判定请求
- `POST /v1/noun`：二元判断快捷端点（`{"rubrics":"...","context":"..."}`）
- `POST /v1/choose`：多分类快捷端点（`{"rubrics":"...","context":"...","choices":"..."}`）
- `POST /v1/score`：分级打分快捷端点（`{"rubrics":"...","context":"...","levels":"..."}`）
- `GET /health`：连通性与健康状态探测

### 5. `decision-cli kill` — 扫描并终止所有相关 MLX 进程，释放统一内存

扫描 macOS 进程表，自动识别并列出所有相关的 MLX 推理后台进程，彻底释放统一内存并清理残留 Socket：

```bash
# 扫描并优雅终止所有关联的 MLX 守护进程：
decision-cli kill

# 仅查看当前运行中的相关 MLX 进程（不执行终止）：
decision-cli kill --dry-run

# 强制强杀进程（SIGKILL）：
decision-cli kill --force
```

---

## 五、退出码与管道调用规范

- `0`：执行成功并正常返回判定结果（无论结果是「是」或「否」）。
- `1`：参数校验错误（如必填命名参数缺失、格式错误等）。
- `2`：运行时或模型加载故障（如 MLX 环境缺失、模型权重未找到等）。
- `3`：文件读写或解析错误。

**管道与 Shell 调用示例**：
```bash
# 获取二元纯标签：
RESULT=$(decision-cli noun --rubrics "是否垃圾评论" --context "兼职刷单月入过万加微..." --format label-only)
if [ "$RESULT" = "是" ]; then
    echo "拦截垃圾内容"
fi

# 使用 jq 提取概率值：
PROB=$(decision-cli noun --rubrics "是否包含技术疑问" --context "请问 Go 协程池怎么写" | jq '.probability')
echo "技术问题概率: $PROB"
```

---

## 六、数据清理与卸载

系统提供内置命令用于停止后台常驻进程并清理本机生成的数据：
```bash
# 终止后台 Worker 与状态栏，清理 ~/.cache/decision-cli 下的虚拟环境与 SQLite 缓存
decision-cli uninstall

# 非交互式执行，并一并移除全局配置文件 ~/.config/decision-cli
decision-cli uninstall --yes --all
```
若需完全删除编译的二进制文件：
```bash
rm -f $(which decision-cli) bin/decision-cli
```
