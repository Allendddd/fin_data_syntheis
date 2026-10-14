# v3 SFT 数据流程

`config/mint-synth.example.yaml` 是 v3 示例。已有 v2 配置可用 `mint-synth migrate-v2 --input old.yaml --output new.yaml --cutoff YYYY-MM-DD` 迁移。旧计划必须继续使用创建时的代码版本；代码指纹门控不允许直接换实现续跑。

## 选择合成范围

`query_export.enabled` 控制长程 query 输出；`training.enabled` 控制 SFT 数据准备，`training.atomic_enabled` 与 `training.long_enabled` 控制 SFT 任务族。配置名 `training` 只涉及数据生成，项目不执行参数训练。`capabilities`、`financial_topics`、`objective_types`、`formula_ids`、`seed_ids` 是白名单，空列表表示不限；长程 query 和 SFT 使用相同的标签筛选。`financial_topic` 从证据指标映射为 `revenue`、`profitability`、`cash_flow`、`balance_sheet`、`valuation` 或 `other`。

长任务在图规划前按种子目标类型和公式 ID 收窄候选；定制种子找不到匹配目标时不会静默回退。只有明确设置 `training.seed_fallback: true` 才补充内置种子。源文档的 `input.available_at_field` 必须列入 `input.metadata_fields`；晚于 `run.cutoff` 的片段不会进入长任务或检索索引。新运行强制输入文件 SHA-256 指纹，以便续跑时检测内容变化。

自定义种子通过 `query_realization.seed_file` 指向 JSONL，每条提供实体无关的任务规格模板：

```json
{"seed_id":"margin-change","seed_query":"比较同一主体两个期间的指标变化。","task_spec_template":{"objective_type":"change","reasoning_form":"retrieve compatible values for two periods and compute change","required_fact_roles":["new-period value","old-period value"],"time_sensitive":false,"requested_output":"direction and magnitude of change"}}
```

## 长程 query 与教师轨迹

`corpus_index` 冻结本地文本分块，`retrieval_index` 按规范化实体桶建立 SQLite FTS5 检索分区。`query_export` 读取验证通过的长任务，输出纯 query 与独立的隐藏校验文件。它依赖验证与检索索引，不依赖 `sft_prepare`。因此教师 rollout 没有被接纳时，已完成的 query 仍可从 query 阶段取得；格式见 [长程 query 文档](opd.md)。

启用 `training.enabled` 后，`sft_prepare` 按验证任务分片运行。可在 `llm.roles.teacher` 单独配置强模型并令 `training.teacher_role: teacher`；原子任务使用已审核的推导轨迹。长任务的强教师只获得 query、截止日和默认的 `search_corpus`、`open_chunk`、`calculate` 工具；可通过 `harness.tool_registry` 扩展。模型必须先搜索再打开证据块。教师轨迹与外部 OPD 使用同一 [Harness 会话](harness.md)：只有必需来源证据能在快照中重放、最终答案与确定性计算一致且动作有效，轨迹才被接纳。每个长任务最多尝试 `rollouts_per_task` 次，每次最多 `max_tool_steps` 次工具调用，并另有一次最终提交机会。无关工具调用被屏蔽损失，工具观察不参与损失。

发布目录 `stages/export/shards/000000/sft/` 包含 `atomic-train.jsonl`、`atomic-val.jsonl`、`long-train.jsonl`、`long-val.jsonl`。每行是 ms-swift `messages` 数据；长任务另有 JSON 字符串 `tools`，包含 `tool_call` 和 `tool_response` 消息。`audit-sidecar.jsonl` 按任务保存标签、答案、种子、公式和证据，这些字段不进入模型可见消息。划分由任务 ID 的哈希确定。

长任务样本结构示意：

```json
{"messages":[{"role":"system","content":"...","loss":false},{"role":"user","content":"...","loss":false},{"role":"tool_call","content":"{\"name\":\"search_corpus\",\"arguments\":{\"query\":\"...\"}}","loss":true},{"role":"tool_response","content":"{\"hits\":[...]}","loss":false},{"role":"assistant","content":"...","loss":true}],"tools":"[{\"type\":\"function\",...}]"}
```

原始审计任务仍可按 `output.export_formats` 输出 Parquet/JSONL。启用的 SFT 任务族若训练集样本少于相应 `min_*_samples`，最终 `export` 不会发布 `_PUBLISHED.json`；独立完成的 query 阶段产物仍保留。`mint-synth status --run-dir ...` 可查看分片，`mint-synth run --run-dir ...` 会执行未完成分片。
