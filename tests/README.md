# decision-cli 功能与集成测试

本目录包含 `decision-cli` 的端到端功能验证脚本与静态测试固件。

---

## 1. 目录结构

```text
tests/
├── README.md             # 测试运行说明
├── fixtures/             # 测试静态固件 (样本数据与多模态测试图片)
│   ├── sample_batch.json
│   ├── red_alert.png
│   └── blue_circle.png
├── common.py             # 测试公共工具函数与 CLI 调用封装
├── run_tests.py          # 自动化测试发现与执行入口
├── test_inference.py     # 单条二元判定 (noun)、多分类 (choose) 与分级打分 (score)
├── test_batch.py         # 批量文件处理与过滤管道 (batch)
├── test_server.py        # HTTP API 服务 (/health, /v1/noun, /v1/systemone) 与守护进程管理 (kill)
└── test_multimodal.py    # 视觉多模态单图判定与双图对比选择 (--image)
```

---

## 2. 运行测试

### 2.1 运行全量 Python 集成测试

```bash
# 运行全部集成测试
python3 tests/run_tests.py

# 或通过 Python 内置 unittest 运行单项测试
python3 -m unittest tests/test_inference.py
python3 -m unittest tests/test_batch.py
python3 -m unittest tests/test_server.py
python3 -m unittest tests/test_multimodal.py
```

> [!NOTE]
> 测试脚本默认优先调用 `bin/decision-cli`，若不存在则回退至 `go run ./cmd/decision-cli`。也可以通过环境变量 `DECISION_CLI_BIN` 指定二进制路径。

### 2.2 运行 Go 原生单元测试

```bash
go test -v ./...
```
