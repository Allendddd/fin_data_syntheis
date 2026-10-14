# 长程 query 数据与外部 OPD

## 合成顺序

本项目先从语料和任务种子构造任务规格，选择有截止日期的来源证据，组成可执行 evidence DAG；确定性回放固定唯一答案后，才生成自然语言 query 并通过语义审计。验证通过的任务形成长程 query 数据。具体构造约束见 [方法文档](methodology.md)。

SFT 数据准备是可选的后续步骤：强模型只看到 query、截止日期和 harness 工具，在冻结语料快照中执行检索与计算。只有答案与隐藏答案图一致、必需证据块已打开的轨迹才保留为 `messages` 样本。query 导出是独立阶段，不依赖教师轨迹是否被接纳。

论文的 OPD 阶段使用 query 驱动学生在训练过程中执行并生成失败轨迹。本项目交付 query、隐藏校验资料、快照索引、可选的教师 SFT 轨迹，以及教师和外部 OPD 可共用的 [有状态 harness](harness.md)。学生策略调用、失败轨迹的关键步骤判别、逐 token 打分和参数更新由项目外的训练系统承担。

## 文件与字段

query 阶段完成后即可读取：

```text
<run_dir>/stages/query_export/shards/000000/queries/
  train.jsonl
  val.jsonl
  hidden-verifier.jsonl
  manifest.json
```

最终 `export` 成功后，这组文件还会复制到 `stages/export/shards/000000/queries/` 并进入发布校验清单。`train.jsonl`、`val.jsonl` 每行仅有模型可见字段：

```json
{"query":"比较某公司两个期间的指标变化，并给出依据。"}
```

`hidden-verifier.jsonl` 每行包含 `query_id`、`split`、分组内从 0 开始的 `line_index`、`query_sha256`、相对运行目录的 `snapshot_index`，以及完整隐藏 `task`。其中 `task` 提供答案、证据、DAG、截止日等校验信息。外部系统可按 `(split, line_index)` 对应 query 与隐藏契约。不能把 `hidden-verifier.jsonl` 拼入模型输入；外部 harness 只应提供 query、工具定义和学生可见的工具观察。快照路径需相对于同一个运行目录解析。

`manifest.json` 记录 train/val 数量、模型可见字段、截止日、工具集指纹，以及三份 JSONL 和各快照的大小与 SHA-256。`mint_synth.harness.iter_cases(run_dir, split="train")` 会检查这些绑定并返回环境端使用的 `HarnessCase`。train 数量小于 `query_export.min_train_queries` 时，query 阶段失败并保留失败记录。拆分由任务 ID 的哈希确定；同一任务续跑时 train/val 归属不变。

## 只合成 query

将配置设为：

```yaml
query_export: {enabled: true, min_train_queries: 1}
training:
  enabled: false
atomic:
  enabled: false
```

其余 `training` 字段如 `financial_topics`、`objective_types`、`formula_ids`、`seed_ids` 仍作为 query 标签筛选配置，`split_validation_fraction` 仍决定划分比例。不会安排 `sft_prepare` 阶段。运行和恢复命令见 [README](../README.md)。
