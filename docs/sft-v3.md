# v3 SFT 数据流程

`config/mint-synth.example.yaml` 是新运行的 v3 示例。已有 v2 配置可用 `mint-synth migrate-v2 --input old.yaml --output new.yaml --cutoff YYYY-MM-DD` 生成 v3 配置。已经冻结的 v2 运行如果要继续执行，仍需使用其计划记录的旧版代码；代码指纹门控不会允许更换实现后直接续跑。

## 选择合成范围

`training` 中的 `atomic_enabled`、`long_enabled` 控制两个任务族。`capabilities`、`financial_topics`、`objective_types`、`formula_ids`、`seed_ids` 是白名单；空列表表示不限。`financial_topic` 从证据中的财务指标映射为 `revenue`、`profitability`、`cash_flow`、`balance_sheet`、`valuation` 或 `other`。标签同时写入审计侧车文件。

长任务在图规划前按种子目标类型和公式 ID 收窄候选，定制种子找不到匹配目标时不静默回退。只有明确设置 `seed_fallback: true` 才补充内置种子。源文档的 `input.available_at_field` 必须通过 `input.metadata_fields` 保留；如还提供 `published_at`，取较晚日期。晚于 `run.cutoff` 的片段不会进入检索索引或长任务；时间敏感任务的证据必须有合法可用日期。新运行强制输入文件的 SHA256 指纹，以便续跑时检测源文件内容变化。

## 训练样本

`corpus_index` 冻结本地完整文本分块，`retrieval_index` 按规范化实体桶并行建立 SQLite FTS5 检索分区。长任务只检索其实体对应的分区；时间敏感任务的检索工具只返回有合法可用日期的片段。`sft_prepare` 依照验证任务分片续跑。原子任务采用已审核的推导轨迹；长任务的教师仅获得问题、截止日和三种工具：`search_corpus`、`open_chunk`、`calculate`。模型必须先搜索再打开证据块。系统会检查所需证据块都被打开、最终答案与确定性计算结果吻合，才接纳轨迹。每个长任务最多尝试 `rollouts_per_task` 次，单次最多 `max_tool_steps` 步。工具观察不参与损失。

发布目录 `stages/export/shards/000000/sft/` 包含 `atomic-train.jsonl`、`atomic-val.jsonl`、`long-train.jsonl`、`long-val.jsonl`。每行是 ms-swift `messages` 数据；长任务另有 JSON 字符串 `tools`，包含 `tool_call` 和 `tool_response` 消息。`audit-sidecar.jsonl` 按 `task_id` 保存标签、答案、种子、公式和证据 ID；这些字段不进入模型可见消息。分割由任务 ID 的哈希确定，同一任务续跑时分割不变。

长任务单行的结构如下（内容省略）：

```json
{"messages":[{"role":"system","content":"...","loss":false},{"role":"user","content":"...","loss":false},{"role":"tool_call","content":"{\"name\":\"search_corpus\",\"arguments\":{\"query\":\"...\"}}","loss":true},{"role":"tool_response","content":"{\"hits\":[...]}","loss":false},{"role":"assistant","content":"...","loss":true}],"tools":"[{\"type\":\"function\",...}]"}
```

原始审计任务仍可按 `output.export_formats` 输出 Parquet/JSONL。任何已启用任务族的训练集样本少于相应 `min_*_samples`（至少 1 条）时，导出阶段会失败，不发布 `_PUBLISHED.json`。`mint-synth status --run-dir ...` 可查看分片完成情况，`mint-synth resume --run-dir ...` 只提交未完成分片。
