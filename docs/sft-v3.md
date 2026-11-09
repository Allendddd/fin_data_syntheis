# v3 SFT 数据流程

`config/mint-synth.example.yaml` 是 v3 示例。已有 v2 配置可用 `mint-synth migrate-v2 --input old.yaml --output new.yaml --cutoff YYYY-MM-DD` 迁移。旧计划必须继续使用创建时的代码版本；代码指纹门控不允许直接换实现续跑。

## 选择合成范围

`query_export.enabled` 控制长程 query 输出；`training.enabled` 控制 SFT 数据准备，`training.atomic_enabled` 与 `training.long_enabled` 控制 SFT 任务族。配置名 `training` 只涉及数据生成，项目不执行参数训练。`capabilities`、`financial_topics`、`objective_types`、`formula_ids`、`seed_ids` 是白名单，空列表表示不限；长程 query 和 SFT 使用相同的标签筛选。`financial_topic` 从证据指标映射为 `revenue`、`profitability`、`cash_flow`、`balance_sheet`、`valuation` 或 `other`。

长任务逐个种子编译 `objective_type`、`reasoning_form`、`required_fact_roles`，先据此限制公式和事实选择，再规划图；无法映射的自定义推导形式或角色在预检时报错。内置 `comparison` 种子目前没有可执行公式，不产出该类图。深度或来源宽度不足的图不会因为匹配种子而被放行。只有明确设置 `training.seed_fallback: true` 才补充内置种子。源文档的 `input.available_at_field` 必须列入 `input.metadata_fields`；晚于 `run.cutoff` 的片段不会进入长任务或检索索引。新运行强制输入文件 SHA-256 指纹，以便续跑时检测内容变化。

同一实体中，多个种子若编译出相同的公式集合与时间过滤条件，只执行一次规则图规划；各个种子仍分别检查候选及其发布约束。分片指标 `planner_cache_hits` 记录避免的重复规划次数。

自定义种子通过 `query_realization.seed_file` 指向 JSONL，每条提供实体无关的任务规格模板：

```json
{"seed_id":"margin-change","seed_query":"比较同一主体两个期间的指标变化。","task_spec_template":{"objective_type":"change","reasoning_form":"retrieve compatible values for two periods and compute change","required_fact_roles":["new-period value","old-period value"],"time_sensitive":false,"requested_output":"direction and magnitude of change"}}
```

## 长程 query 与教师轨迹

`corpus_index` 冻结本地文本分块，`retrieval_index` 按规范化实体桶建立 SQLite FTS5 检索分区。`query_export` 读取验证通过的长任务，输出纯 query 与独立的隐藏校验文件。它依赖验证与检索索引，不依赖 `sft_prepare`。因此教师 rollout 没有被接纳时，已完成的 query 仍可从 query 阶段取得；格式见 [长程 query 文档](opd.md)。

启用 `training.enabled` 后，`sft_prepare` 按验证任务分片运行。原子任务默认由 `training.atomic_teacher_roles: [constructor]` 中的角色各采样 `atomic_samples_per_task: 2` 次；教师提示词只含上下文和问题。候选必须给出逐字证据引用、完整结构化运算和类型化答案，私有校验器将其映射到已审核答案图；`atomic_selection_policy: concise` 选取最短合格 JSON，`first_valid` 按角色和尝试顺序选取。所有候选失败默认拒绝；只有 `atomic_audit_fallback: true` 才使用单独标记的原审计轨迹。可将角色改为独立的 `llm.roles.teacher`。

每个原子候选和长任务 rollout 使用稳定的采样标识区分缓存；断点续跑可重放同一标识的结果。启用 `llm.endpoints.*.send_seed` 时，客户端从运行 seed 和采样标识派生不同的请求 seed，标识本身不会写入教师提示词。多次采样仍受模型温度和服务端实现影响，不保证候选一定不同。

`sft_prepare` 在每个分片内以 `llm.concurrency_per_worker` 为上限并发处理任务，输出顺序保持与验证任务一致。整个本地运行的模型请求上限约为 `mint-synth run --workers` 与该值的乘积；实际吞吐还取决于服务端限流和长任务每条轨迹的顺序工具调用。各分片 `_SUCCESS.json` 的 `metrics.processing_seconds` 记录处理耗时，可结合候选数、接纳数和 LLM 用量计算吞吐；先用小规模样本观察这些指标，再提高并发与采样次数。

长任务的 `long_teacher_roles` 默认空列表，此时沿用旧字段 `teacher_role`；非空时按 `rollouts_per_task` 次尝试轮换角色。教师只获得 query、截止日和默认的 `search_corpus`、`open_chunk`、`calculate` 工具；可通过 `harness.tool_registry` 扩展。模型必须先搜索再打开证据块。教师轨迹与外部 OPD 使用同一 [Harness 会话](harness.md)：必须将已取得证据绑定到最终证明中的事实，并完整表达必要运算，答案和单位与隐藏图一致且动作有效，轨迹才被接纳。每次最多 `max_tool_steps` 次工具调用，并另有一次最终提交机会。无关工具调用被屏蔽损失，工具观察不参与损失。

发布目录 `stages/export/shards/000000/sft/` 包含 `atomic-train.jsonl`、`atomic-val.jsonl`、`long-train.jsonl`、`long-val.jsonl`。每行是 ms-swift `messages` 数据；长任务另有 JSON 字符串 `tools`，包含 `tool_call` 和 `tool_response` 消息。`audit-sidecar.jsonl` 按任务保存标签、答案、种子、公式、证据，以及教师候选来源、拒绝理由和选择依据；这些字段不进入模型可见消息。划分由任务 ID 的哈希确定。

长任务样本结构示意：

```json
{"messages":[{"role":"system","content":"...","loss":false},{"role":"user","content":"...","loss":false},{"role":"tool_call","content":"{\"name\":\"search_corpus\",\"arguments\":{\"query\":\"...\"}}","loss":true},{"role":"tool_response","content":"{\"hits\":[...]}","loss":false},{"role":"assistant","content":"...","loss":true}],"tools":"[{\"type\":\"function\",...}]"}
```

原始审计任务仍可按 `output.export_formats` 输出 Parquet/JSONL。启用的 SFT 任务族若训练集样本少于相应 `min_*_samples`，最终 `export` 不会发布 `_PUBLISHED.json`；独立完成的 query 阶段产物仍保留。`mint-synth status --run-dir ...` 可查看分片，`mint-synth run --run-dir ...` 会执行未完成分片。
