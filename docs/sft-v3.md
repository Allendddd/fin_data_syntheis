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

启用 `training.enabled` 后，`sft_prepare` 按验证任务分片运行。原子教师只接收 context 与 question，各采样 `atomic_samples_per_task: 2` 次自然语言推导和答案；不规定 proof JSON、逐字引用、结束标记或参考图路径。真实可见 CoT 与 content 进入监督目标，完整 exchange 和原始字段留在审计记录中。

原子文本 completion 使用独立的模式、版本及完整 exchange 缓存，即使共享 endpoint 的 `strict_json=true` 也不发送 JSON response_format，不解析解答为 JSON。缓存覆盖模型/revision、有效参数、prompt 和 sampling key，不命中旧 proof/parsed-only 缓存；不同 attempt 分离，同一 key 可恢复。内部构造与审计仍使用 JSON。

`sft_prepare` 在每个分片内以 `llm.concurrency_per_worker` 为上限并发处理任务，输出顺序保持与验证任务一致。整个本地运行的模型请求上限约为 `mint-synth run --workers` 与该值的乘积；实际吞吐还取决于服务端限流和长任务每条轨迹的顺序工具调用。各分片 `_SUCCESS.json` 的 `metrics.processing_seconds` 记录处理耗时，可结合候选数、接纳数和 LLM 用量计算吞吐；先用小规模样本观察这些指标，再提高并发与采样次数。

长任务的 `long_teacher_roles` 默认沿用 `teacher_role`。项目 JSON runtime 的 `final` 承载自然语言；finance/MCP runtime 使用 `submit_final(answer: string)`；框架原生工具模式使用正常最终回复。`long_verifier_role` 默认 `auditor`，在采集冻结后审核真实可见推理、最终答案和实际观察，并回放候选自身的运算。可恢复错误留作观察并允许继续；合格轨迹中错误、重复和无贡献动作及关联文字屏蔽 loss。

发布目录 `stages/export/shards/000000/sft/` 包含 `atomic-train.jsonl`、`atomic-val.jsonl`、`long-train.jsonl`、`long-val.jsonl`。每行是 ms-swift `messages` 数据；长任务另有 JSON 字符串 `tools`，包含 `tool_call` 和 `tool_response` 消息。`audit-sidecar.jsonl` 按任务保存标签、答案、种子、公式、证据，以及教师候选来源、拒绝理由和选择依据；这些字段不进入模型可见消息。划分由任务 ID 的哈希确定。

Strands、Codex、Claude Code 的 `mode: native` 与项目自有 JSON action `default: native` 区分。原生模式使用框架工具，不注入金融 MCP 或 `submit_final`，正常回复不要求 proof JSON。实际读取、源文件完整性、时间边界和内容审核通过后按真实协议导出；原生 loop 期限由 `agent_runtime.timeout_seconds` 控制。

原生轨迹按实际模型请求写入独立文件，不混入上述 ms-swift 数据：CC 使用 `cc-native-{train,val}.jsonl` / `claude_code_native_v2`，保留 Anthropic system 与内容块；Codex 使用 `codex-native-{train,val}.jsonl` / `codex_native_v2`，保留 Responses `input`、工具和完整 `output`；Strands 使用 `strands-native-{train,val}.jsonl` / `strands_native_v2`，保留 Chat Completions 请求和实际 assistant message。原始工具名、调用 ID、schema 与该次上下文均保留。字段及监督范围见 [原生工具模式](teacher-runtimes.md#三种框架的原生工具模式)、[CC 契约](teacher-runtimes.md#原生-cc-训练数据契约) 和 [Codex / Strands 契约](teacher-runtimes.md#codex-与-strands-原生训练数据契约)。

同一任务的所有原生请求属于同一 split；`sample_count` 是请求数，最低长轨迹门槛仍按合格完整轨迹数计，不能用一条轨迹的多个请求放大样本门槛。原生协议文件需要训练端按学生模板适配，不能直接视为统一金融 messages。

长任务样本结构示意：

```json
{"trajectory_id":"...","messages":[{"role":"system","content":"...","loss":false},{"role":"user","content":"...","loss":false},{"role":"assistant","content":"<教师实际返回的分析或规划文本>","loss":true},{"role":"assistant","content":"<memory>{\"plan\":\"<教师实际返回的工作计划>\"}</memory>","loss":true},{"role":"tool_call","content":"{\"name\":\"search_corpus\",\"arguments\":{\"query\":\"...\"}}","loss":true},{"role":"tool_response","content":"{\"hits\":[...]}","loss":false},{"role":"assistant","content":"<教师实际返回的后续分析>","loss":true},{"role":"assistant","content":"<教师实际生成的自然语言推导与答案>","loss":true}],"tools":"[{\"type\":\"function\",...}]"}
```

`tool_call` 是实际模型调用，`tool_response` 是真实环境返回；通过 [ms-swift agent template](https://swift.readthedocs.io/en/v4.6/Instruction/Agent-support.html) 转换为学生格式。实际可见推理、规划、反思和摘要原样保留；没有返回时不补写，不将摘要称作完整 CoT。最终 assistant 是真实自然语言解答。

`training.reasoning_format` 默认 `plain`，将接口暴露的推理原文作为普通 assistant 文本；设为 `think` 时，为独立推理字段增加一层 `<think>…</think>`，已有该标记的原文不重复包裹。推理摘要仍作为摘要文本保留，不能声称是未公开的内部推理。应按学生模型的 chat/agent template 选择设置；本项目不自动猜测目标训练模板，也未依赖 ms-swift 执行训练。

实际 `memory` 更新保留为 `<memory>JSON</memory>` 工作笔记，不作为来源证据。模型文字、笔记和调用按相关性设置 loss；混合有效与无效动作的一轮，其共享推理保守屏蔽，各调用单独标注。ms-swift 合并相邻调用会使用首个 loss，无法表达的混合 mask 明确拒绝，不能静默训练无效动作。

项目 `native` 的真实请求仍是 system 与公开状态 JSON，模型通过 action 包络操作环境。归一化 messages 先导出实际模型内容和调用，再导出观察，最终自然语言与实际提交绑定。采集、执行对不上，或 final 与尚未观察的调用同批生成时明确拒绝，保留原始记录。

查看原始交互：先用样本的 `trajectory_id` 在 `sft/trajectory-index.jsonl` 中查到候选 `trace`。该目录的 `trajectory.json` 提供 `message_origins`，把每条导出内容绑定到模型 call ID、原始文件、来源字段和环境步骤。native 的 `model/calls/000001.json` 等文件保存完整请求和响应；`model.parsed.json` 仅是兼容的解析摘要。外部 runtime 的模型请求、SSE 和 SDK/CLI 事件见 [教师 runtimes](teacher-runtimes.md#轨迹产物)。缺失完整模型记录或遇到无法解释的内容类型时，禁止退回工具轨迹导出。

代码指纹及默认配置已变化，需要新建运行计划。历史样本不会自动改写；只有解析动作的旧产物无法恢复未保存的分析，需要重新采集。

原始审计任务仍可按 `output.export_formats` 输出 Parquet/JSONL。启用的 SFT 任务族若训练集样本少于相应 `min_*_samples`，最终 `export` 不会发布 `_PUBLISHED.json`；独立完成的 query 阶段产物仍保留。`mint-synth status --run-dir ...` 可查看分片，`mint-synth run --run-dir ...` 会执行未完成分片。

## 内容审核、参考推导与格式迁移

`atomic_verifier_role` 和 `atomic_judge_role` 默认 `auditor`。私有审核可读取隐藏参考，但不能把它注入教师或替教师补齐推导。审核器提取候选实际引文和计算，本地核验来源、显式数值/维度和计算结果；接受等价运算及路径。知识、抽取和文本分析按实际需要语义审核，不强行增加算术步骤。语义等价、期间解释与推导提取依赖 LLM，存在误判，不是完全确定性的自由文本证明器；无法可靠验证时拒绝。

`atomic_selection_policy: judge` 按完整性、金融正确和清晰度择优。`concise` 是记录迁移信息的弃用别名，不再按长度选择；`first_valid` 是显式非论文模式。单个合格候选直接选择，多候选 judge 失败拒绝教师候选选择，不回退。

`atomic_include_reference: true` 默认独立纳入已审核并通过内容复核的原始 r；`atomic_reference_fraction: 1.0` 按 task ID 确定性抽样。默认最多一个教师样本加一个参考样本，这是可配置工程配方，不是论文公开比例。缺失或不合格的 r 跳过留因，同任务样本共享 split，以 sample_id 关联来源。兼容的 atomic_audit_fallback 与独立参考来源分别记账，不能伪装成教师输出。

每个教师候选增加一次审核，多合格候选另加 judge，参考 r 另做一次复核；长程一次审核同时给出动作贡献。审核 JSON、候选图和评委输出只进私有记录。

原生格式升级到 claude_code_native_v2、codex_native_v2、strands_native_v2。CC 消费 target_block_loss_mask；Codex 消费 target_output_loss_mask 和 target_reasoning_text_fields（encrypted_content=false）；Strands 同时消费 target_text_loss 和按调用 ID 的 target_tool_call_loss_mask。训练端必须应用这些字段，不能按 v1 假设整轮 loss=true。输入和观察不训练，可见推理/摘要按来源保留，opaque/redacted 内容只留档。

代码、prompt 和格式变化后需要新 plan/run-dir，不能绕过指纹续跑旧计划。目标 GPU、SDK/CLI 和训练模板的真实 token labels 仍须在目标环境验收。
