# 01 - CLI 接口契约、子命令架构与 Agent 自省规范

## 1. 设计原则

1. **子命令职责分层**：
   - 单条文本判定：`noun`（二元判定）、`score`（分级量规）、`choose`（多分类选择）仅接受文本字符串输入；
   - 批量文件处理：收敛在 `batch` 子命令组（`batch noun`、`batch score`、`batch choose`），仅接受数据文件；
   - 服务与维护：`systemone`（Jev 复合问卷）、`serve`（HTTP API）、`kill`（终止进程）、`doctor`（自检）、`models`（模型管理）、`cache`（缓存管理）。
2. **长参数命名规范**：
   - 统一采用显式长参数（如 `--rubrics`、`--context`、`--choices`、`--levels`、`--input`、`--output`、`--format`、`--model`），避免拼写歧义。
3. **输出通道分离**：
   - stdout 仅输出判定结果（JSON/YAML/文本）；日志、加载进度与警告统一输出至 stderr。

---

## 2. 命令行参数契约

### 2.1 全局公共参数
- `--config <path>`：全局配置文件路径（缺省为 `~/.config/decision-cli/config.yaml`）。
- `--usage`：输出面向 AI Agent 与开发者的结构化集成手册。

### 2.2 单条文本判定子命令

#### 1. `decision-cli noun`（二元判定）
- `--rubrics <string>`：**[必填]** 判定准则。
- `--context <string>`：**[必填]** 待评估文本。
- `--model <string>`：模型别名或路径（默认：`"gemma-4-26b-a4b"`）。
- `--threshold <float>`：正向判定阈值（默认 `0.5`）。
- `--temperature <float>`：校准温度。
- `--format <string>`：输出格式，支持 `json` (默认) | `yaml` | `compact` | `label-only`。
- `--output <path>`：输出保存路径。
- `--no-cache` / `--cache-only`：绕过缓存 / 仅读缓存。

#### 2. `decision-cli score`（分级量规打分）
- `--rubrics <string>`：**[必填]** 评分准则。
- `--context <string>`：**[必填]** 待评估文本。
- `--levels <string>`：**[必填]** 逗号分隔的升序刻度列表（例：`"低,中,高"`）。
- `--debias-order`：开启正反序双向前向推理以消除位置偏置。
- `--model`、`--temperature`、`--format`、`--output`、`--no-cache`、`--cache-only`。

#### 3. `decision-cli choose`（互斥多分类选择）
- `--rubrics <string>`：**[必填]** 分类指导准则。
- `--context <string>`：**[必填]** 待评估文本。
- `--choices <string>`：**[必填]** 选项列表，逗号分隔，冒号分隔 `键:描述`（例：`"bug:系统报错,feat:功能建议"`）。
- `--debias-order`：开启正反序双向推理去偏置。
- `--model`、`--temperature`、`--format`、`--output`、`--no-cache`、`--cache-only`。

### 2.3 批量文件处理子命令组 (`decision-cli batch`)

批量命令仅接受文件输入（`--input`），禁止传入 `--context`。

#### 通用参数：
- `--input <path>`：**[必填]** 批量输入文件（`.json`, `.jsonl`, `.yaml`）。
- `--output <path>`：输出文件路径（缺省输出至 stdout）。
- `--format <string>`：输出格式（`json` | `yaml` | `jsonl`）。
- `--id-field <string>`：记录唯一标识字段名（默认 `"id"`）。
- `--context-template <string>`：多字段拼接 Go 模板。
- `--include-record`：输出中附带原始记录数据。
- `--progress`：向 stderr 实时打印处理进度。

#### 子命令扩展：
- `batch noun`：支持 `--threshold` 与 `--filter-passed`（仅输出正向记录）。
- `batch score`：支持 `--levels` 与 `--debias-order`。
- `batch choose`：支持 `--choices` 与 `--debias-order`。

### 2.4 服务与维护子命令
- `decision-cli systemone --input <path_or_json>`：处理 Jev 规范复合问卷。
- `decision-cli serve [--host 127.0.0.1] [--port 8080] [--model gemma-4-26b-a4b]`：启动 HTTP API 服务。
- `decision-cli kill [--force] [--dry-run]`：清理残留 Worker 进程。
- `decision-cli doctor [--model <name>]`：系统环境与基准测试自检。
- `decision-cli models [--format table|json|yaml]`：查看就绪与支持模型。
- `decision-cli cache [stats|export|clear]`：本地 SQLite 缓存运维。

---

## 3. 退出码契约

| 退出码 | 含义 | 说明 |
| :---: | :--- | :--- |
| `0` | 执行成功 | 标准输出返回请求的结构化数据 |
| `1` | 业务不匹配 / 缓存未命中 | `--cache-only` 未命中缓存，或条件判定未通过 |
| `2` | 参数错误 | 缺少必需参数或输入格式不合法 |
| `3` | 运行环境异常 | Python/MLX 运行时损坏或硬件不支持 |
| `4` | 模型路径异常 | 模型未找到或权重损坏 |
