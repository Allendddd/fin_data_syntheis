# Mint-Agent 数据方法与 v2 工程映射

本文记录基础任务构造、证据 DAG 与确定性审计。v3 的教师轨迹见 [SFT 数据流程](sft-v3.md)，长程 query 导出及外部 OPD 数据契约见 [长程 query](opd.md)。

## 1. 复现目标

Mint-Agent 的公开方法把训练数据分为两类：

- `D_atom`：上下文已给定，训练知识、抽取、计算、分析、验证五类边界明确的金融能力。
- `D_long`：不给上下文，训练 agent 检索多个来源并完成长链金融研究任务。

两类数据共享一个关键约束：答案必须来自可定位证据，并且推导能被执行器重新计算。长链数据尤其强调“先答案图、后问题”：先从多来源事实形成可执行 evidence DAG，固定唯一答案后，才把图实现为自然语言研究问题。

本项目复现的是论文公开的 evidence contract 和构造顺序，不宣称还原未公开的内部数据集、prompt、seed bank 或 MintHarness。

## 2. 输入适配

论文使用披露文档、结构化财务数据和稳定网页快照。用户输入只有统一 `text_field` 时，v2 做如下适配：

1. JSONL 文件按字节范围规划，worker 在换行处对齐；Parquet 按 row group 规划。
2. 每条记录保存文件 URI、格式、字节区间或 row group/row index。
3. 可选 `id_field` 先按 ID 独立 hash shuffle；一对多正文冲突会跨所有内容分区全局发现并隔离。
4. 正文 SHA-256 随后在 LLM 调用前用于全局精确去重。
5. 提取事实保留 entity、period、metric、value、unit、scale、currency、逐字 quote 和正文字符区间。

仅靠正文无法恢复原来不存在的 XBRL taxonomy、CIK/LEI 或标准概念映射。因此生产输入应尽量通过 `metadata_fields` 保留 ticker、CIK、LEI、报告期、发布日期、URL 和文档类型。

默认单条正文最多 20,000,000 个 Unicode 字符；JSONL 单行另有 64 MiB 字节上限。该记录级上限不代表模型上下文，模型输入默认按 6,000 字符、400 字符重叠切块。两类限制都可配置，但应与实际推理模型上下文和登录/计算节点内存共同标定。

## 3. 实体解析与 evidence crossover

默认的 crossover 范围是同一 canonical entity 的不同文档。解析顺序为：

1. 配置的强标识符字段；
2. 版本化 alias YAML；
3. Unicode/大小写/标点/公司后缀规范化；
4. 仅对仍有歧义的 block 使用 LLM 聚类回退。

LLM 输出必须构成输入 surface 的完整、无重叠分区，不能引入新名称。同一 surface 若对应多个冲突强标识符，事实会被隔离，避免误合并。解析完成后按 canonical entity 哈希重分区，使一个实体的事实在同一 DAG worker 中汇合。

## 4. DAG 如何形成

公式注册表定义 `formula_id`、版本、generator 和允许的 operation 链。内置生成器包括：

- 同指标跨期间的绝对变化和百分比变化；
- 同实体、同期间的跨来源比率；
- 同指标、同期间、同金融维度的跨来源均值；
- 同指标、同期间的跨来源一致性验证。
- 三个期间、三个事实叶的分支趋势图；
- 两个期间、两组分子/分母的四事实比率变化图；
- 三个独立来源先分别比较、再通过布尔 `and` 汇合的一致性图。

规则规划过程：

1. 只选择带显式数值的事实叶节点。
2. 按公式要求匹配 metric、规范化 period 类型、unit、scale、currency 和来源关系；年度、季度、半年度和点时日期互不混用。
3. 建立只引用现有 `fact_id` 的 fact nodes。
4. 按注册公式建立 operation nodes，并写入公式 ID/版本。
5. 用 Decimal 执行 DAG；除零、循环、未知节点、维度冲突或非有限值立即拒绝；图中所有节点必须是 terminal 祖先，声明 evidence 必须与实际事实祖先完全一致。
6. 从 terminal 的真实祖先计算操作深度和独立来源数。
7. 对规范图哈希得到稳定 `graph_id`；候选扫描上限与有效 beam width 分离，无效的早期组合不会挤掉后续合法图。

v3 长任务先把分析师种子的目标类型、推导形式与必需事实角色映射到已注册的公式生成器，再以该种子的时间约束筛选事实并规划图。无法映射的自定义规格在运行前拒绝；种子不会替代图深度、来源广度或回放门槛。

深度按 terminal 的真实祖先计算，规则公式不会用 `identity` 或无必要的 `abs` 填充深度。因而默认 `min_depth=2` 会自然排除单次 ratio、average 或 equals；需要这些浅图时必须在配置中明确把最小深度降为 1。

默认深度门是 2–4，但当前内置可执行拓扑的实际深度为 1–3；在默认最小深度下，成品 long-horizon 图实际为 2–3 层。`max_depth=4` 是上限而不是难度承诺，仅提高该值不会生成新拓扑。更深任务必须新增受执行器支持的 generator、公式版本和回放/审计测试，不能只在 YAML 中声明任意算式。

当规则候选不足时，LLM 只能补充带确切 fact IDs 的 `same_metric` 关系。系统验证引用后，仍由相同的类型化公式规划器生成 DAG；LLM 不能直接提交自由算式，也不能改写原事实。

## 5. 题目实现与审计

通过回放的 DAG 已经固定 evidence、推导和答案。constructor 只接收去 locator/内部 ID 的语义摘要，将其实现为自然问题。之后执行两类审计：

- 独立 auditor 检查支持性、唯一性、金融有效性、不泄漏以及未复制 seed。
- 本地确定性门重新回放 DAG，检查声明证据集合、答案维度、深度、来源宽度、数字泄漏、内部 ID/locator 泄漏和评测题近重复。

任何一步失败都写入该阶段 quarantine；不会让 auditor “修复”候选后继续。

## 6. 计划与恢复语义

运行计划在执行前冻结：

- 完整默认值展开后的 YAML；
- 输入文件/segment 指纹和 Parquet manifest；
- 代码与内置 prompt/公式资源指纹；
- 自定义公式、alias、evaluation 文件 SHA-256；
- 带旧 SLURM 字段的 v2 计划中的镜像指纹；新的 v2/v3 配置无需这些字段，v3 计划不记录调度器或解释器设置；
- 模型名、可选 revision、endpoint 配置和随机种子；
- 每个 shard 及每个阶段的预期签名。

`mint-synth run` 按计划顺序执行阶段，在单个阶段内可并行处理分片。每个 shard 在同一文件系统写临时产物，完成后原子 rename；已有不同签名的 shard 不直接覆盖。再次使用 `run --run-dir` 时，仅复用签名吻合且产物完整的 `_SUCCESS.json`。运行目录锁防止两个本地进程或集群作业同时处理同一计划。每个 LLM 角色可使用分片级 SQLite 请求检查点，使失败 shard 在恢复时复用已经成功的请求。SLURM 只通过一个作业脚本调用相同的 `run` 命令；分区、资源和 Python 路径由提交脚本、作业脚本与环境变量指定。

长程 query 导出是独立阶段，依赖验证与检索索引，可先于教师 SFT 轨迹完成；隐藏答案图只供离线验证，不进入模型输入。质量比率按完整上游阶段聚合后再阻断下游，避免“小 shard 中一条脏数据”导致与全局阈值不一致。某阶段的分片失败会停止当前运行，恢复时先补齐该阶段，才能执行下游。发布阶段还要求最低有效任务数，并为最终产物生成可复验的 SHA-256 清单。

## 7. 未公开部分与工程补全

论文未公开原始构造 prompt、模型/解码组合、source-pack mining 算法、`h_min/b_min` 取值、完整 analyst seed bank、训练比例和完整 harness。v2 的公式注册表、实体规则、物理 shuffle、默认阈值和 YAML schema 都属于可审计工程补全。v3 的教师证明 JSON、确定性候选选择和种子规格映射也属于工程补全，并非论文内部格式或评判器的逐字复现。

因此合适的描述是 method-level reproduction 或 faithful reimplementation of published constraints，而不是官方数据集的 exact reproduction。

## 8. 上线验收指标

除吞吐和成本外，建议按实体/来源/时间分层抽样并持续记录：

- 原文定位准确率与 numeric fact 准确率；
- 实体误合并率、漏合并率和强标识冲突率；
- 每个公式的 DAG 构造率、回放通过率与来源广度；
- constructor、auditor 的 terminal error/重试/token 指标；
- 语义审计通过率、确定性二次审计淘汰率；
- 与验证/测试题的近重复率；
- 每阶段数据量、quarantine 原因分布、NFS 吞吐与 inode 数。
