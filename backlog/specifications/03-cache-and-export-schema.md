# 03 - SQLite 内容寻址缓存与数据导出契约

## 1. 核心设计原则

1. **内容寻址与无损重算**：
   - 缓存键由模型名、准则与上下文的标准化哈希决定；
   - 缓存记录原始 Logits（`logit_yes`, `logit_no`）。调整判定阈值 `--threshold` 或校准温度 `--temperature` 时直接读取已缓存 Logits 即时计算，无需重新调用模型推理。
2. **批量事务写入**：
   - 批量前向时采用单个 SQLite 事务写入，保障高吞吐与写入一致性。

---

## 2. 缓存算法与数据表结构

### 2.1 缓存键算法
输入文本进行首尾空白裁剪与换行符归一化（`
` -> `
`）：
```text
raw_key = "v1|" + canonicalModelID + "|" + normalize(rubrics) + "|" + normalize(context)
cache_key = SHA256(raw_key) // 64位十六进制字符串
```

### 2.2 表结构 (`~/.cache/decision-cli/cache.db`)
启用 WAL 模式（`PRAGMA journal_mode=WAL;`）：
```sql
CREATE TABLE IF NOT EXISTS decision_cache (
    cache_key      TEXT PRIMARY KEY,
    model_id       TEXT NOT NULL,
    rubric_hash    TEXT NOT NULL,
    rubrics        TEXT NOT NULL,
    record_id      TEXT DEFAULT '',
    context_hash   TEXT NOT NULL,
    context        TEXT NOT NULL,
    logit_yes      REAL NOT NULL,
    logit_no       REAL NOT NULL,
    logit_margin   REAL NOT NULL,
    probability    REAL NOT NULL,
    latency_ms     REAL NOT NULL,
    created_at     TEXT NOT NULL,
    hit_count      INTEGER DEFAULT 0,
    last_accessed  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decision_model_rubric ON decision_cache (model_id, rubric_hash);
CREATE INDEX IF NOT EXISTS idx_decision_record_id ON decision_cache (record_id);
```

---

## 3. 导出数据格式规范 (JSON / YAML)

支持通过 `--output <path>` 或 `decision-cli cache export` 导出结构化记录：

```yaml
model: "gemma-4-26b-a4b"
rubrics: "判断文本是否包含产品功能建议"
threshold: 0.5
summary:
  total: 100
  passed_count: 12
  rejected_count: 88
  cache_hits: 100
  cache_misses: 0
results:
  - id: "item-001"
    decision: "是"
    passed: true
    probability: 0.9482
    confidence: 0.8964
    logits:
      yes: 18.42
      no: 15.51
      margin: 2.91
    cached: true
    cache_key: "8f31c902..."
    context: "希望增加批量导出的 CSV 格式支持"
```
