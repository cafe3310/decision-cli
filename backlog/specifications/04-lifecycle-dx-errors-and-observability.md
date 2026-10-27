# 04 - 依赖生命周期、错误诊断与服务可观测性规范

## 1. 设计原则

1. **依赖隔离与可重现**：Python 运行时严格收敛于专用目录，不污染系统环境。
2. **完全卸载与清理**：提供一键清理命令与资产分布说明。
3. **报错附带诊断修复建议**：拦截异常时给出具体原因与可执行操作。
4. **服务过程可观测**：常驻 HTTP API 模式下提供标准 Stdio 访问日志与耗时度量。

---

## 2. 资产目录与完全卸载规范

### 2.1 本地资产分布
- 配置文件：`~/.config/decision-cli/`
- 缓存与运行时：`~/.cache/decision-cli/`（含 SQLite 缓存、Socket、自包含 venv）
- 执行二进制：`bin/decision-cli` 或系统安装路径

### 2.2 卸载命令 (`decision-cli uninstall`)
```bash
# 终止后台 Worker 与状态栏，清理 ~/.cache/decision-cli 虚拟环境与 SQLite 缓存
decision-cli uninstall

# 非交互式执行，并一并移除全局配置文件
decision-cli uninstall --yes --all
```

---

## 3. 错误信息呈现与修复建议规范

CLI 检测到异常时，统一按结构化格式输出：
```text
错误: <具体原因简述>
诊断信息: <已探测的环境状态或扫描路径>
建议操作:
  1. <直接修复指令>
  2. <进一步排查命令>
```

### 常见场景建议映射
| 场景 | 错误原因 | 建议操作 |
| :--- | :--- | :--- |
| **模型未找到** | `model "xxx" not found` | 提示运行 `decision-cli models` 查看可用模型，或配置权重路径 |
| **纯文本模型传图** | `模型 xxx 不支持 --image` | 建议改用支持多模态的 `gemma-4-26b-a4b` 或 `gemma-4-e4b` |
| **环境损坏** | `自包含环境校验失败` | 建议运行 `decision-cli setup` 重新初始化 |
| **参数缺失** | `缺少 --rubrics 或 --context` | 给出正确命令格式示例，或提示查看 `--help` |
| **端口冲突** | `bind: address already in use` | 提示更换 `--port` 端口，或排查端口占用 |

---

## 4. HTTP API 服务日志规范 (`decision-cli serve`)

运行时日志统一输出至 stderr，保持 stdout 供重定向：
- **格式**：`[HH:MM:SS] METHOD /path STATUS 耗时ms [模型名] 结果摘要 (来源IP)`
- **示例**：
  ```text
  [01:35:10] POST /v1/noun 200 18.2ms [gemma-4-26b-a4b] 决策: 是 (p=0.985, cache: HIT) (127.0.0.1)
  [01:35:12] POST /v1/noun 400 0.5ms [-] 错误: rubrics 为必填字段 (127.0.0.1)
  [01:35:15] GET  /health  200 0.1ms [-] ok (127.0.0.1)
  ```
