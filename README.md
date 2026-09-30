# mint-synth v2

面向 SLURM + 共享 POSIX/NFS 的金融语料合成流水线。输入可以是一个文件、多个文件或目录树，支持普通 UTF-8 JSONL 与 Parquet；所有算法、资源和质量阈值都来自 YAML 配置。LLM 推理服务独立部署，本工具只调用 OpenAI-compatible `/chat/completions` 接口。

这不是“让模型直接看文本编问答”的脚本。长链任务遵循固定顺序：提取可定位事实，解析同一金融实体，按规则/公式先生成并回放答案 DAG，之后才把 DAG 实现成自然问题，最后由独立 auditor 和确定性校验器共同验收。

## 数据流与 DAG

```text
Input catalog (JSONL byte ranges / Parquet row groups)
  -> normalize
  -> global external-ID validation shuffle
  -> exact-content dedup + quarantine
  -> provenance-linked fact extraction
     +-> atomic task construction/audit
     -> entity resolution (identifier -> alias -> normalized name -> LLM fallback)
        -> canonical-entity hash repartition
           -> typed rule/formula DAG planner + constrained LLM relation supplement
              -> deterministic replay
                 -> query realization + independent semantic audit
                    -> validation -> Parquet / optional JSONL publication
```

DAG 不是 LLM 随手输出的自由 JSON。公式注册表为每个 operation 固定 `formula_id`、版本和受支持 generator 的拓扑；规则规划器只从现有 `fact_id` 组合候选，强制单位/尺度/币种、期间类型、实体及来源广度约束。年度、季度、半年度和日期事实不会被当成同一种时间粒度混算。LLM 仅在规则候选不足时补充 `same_metric` 语义关系，而且只能引用已存在的 `fact_id`。每个候选必须通过 Decimal 确定性回放、最小深度、最小来源数和终点答案一致性检查；声明的 evidence 必须与 terminal 的事实祖先完全一致，断开节点、重复输入、单输入聚合和 `identity` 深度填充都会被拒绝，之后才允许生成问题。

内置公式位于 [formulas.default.yaml](src/mint_synth/resources/formulas.default.yaml)，包括期间绝对/百分比变化、跨来源比率与均值、三期趋势、四事实的跨期比率变化，以及三来源分支验证。生产环境可以用 `dag.formula_registry` 指向自有版本化 YAML；自定义表可以选择、命名和版本化现有的安全 generator，但不能注入未经执行器实现的新拓扑。深度只按真实祖先操作计算，不用 `identity` 填充；因此默认 `min_depth: 2` 会过滤单次 ratio/average/equals，若需要浅图必须显式设为 1。

## 安装与镜像

本地开发使用 Python 3.10–3.13。生产镜像固定为 Python 3.12.14，并在 `Dockerfile` 与 Apptainer definition 中锁定 Docker Official Image 的多架构 digest；升级基础镜像时应同时修改两处并重新执行完整 CI：

```bash
python -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -e '.[dev]'
.venv/bin/python -m unittest discover -s tests -v
```

Windows PowerShell 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。

构建 OCI 镜像：

```bash
docker build -t registry.example.com/mint-synth:2.0.0 .
docker push registry.example.com/mint-synth:2.0.0
apptainer pull deploy/mint-synth.sif docker://registry.example.com/mint-synth:2.0.0
```

也可以在仓库根目录直接构建 SIF：

```bash
apptainer build deploy/mint-synth.sif deploy/apptainer.def
```

Docker 构建可通过 `--build-arg PYTHON_BASE_IMAGE=<registry>/<image>@sha256:<digest>` 使用站点镜像仓库；传入值仍应固定 digest，不能在正式构建中只用浮动 tag。直接构建 Apptainer 时则修改 definition 中的 `From` 并纳入代码审查。

## 配置

完整配置见 [mint-synth.example.yaml](config/mint-synth.example.yaml)，严格 schema 见 [config-v2.schema.json](src/mint_synth/resources/config-v2.schema.json)。未知键会直接报错，默认值会写入每次运行的 `resolved_config.yaml`。CLI 不接受任意 `key=value` 覆盖，避免提交命令与审计配置不一致。

最少需要修改：

- `input.paths`：文件或目录列表；目录默认递归发现 `**/*.jsonl` 和 `**/*.parquet`。
- `input.text_field` / `input.id_field`：全局一致的正文和可选 ID 字段。
- `output.root`、`run.work_dir`：共享存储上的输出和 DuckDB spill 目录。
- `slurm.apptainer_image` / `slurm.bind_paths`：worker 镜像及挂载路径。直接 `sbatch` 模式的账号、分区、QOS、CPU、内存、时限和数组并发写在 `slurm/*.sbatch`；YAML 中同类字段只供兼容的动态生成模式使用。
- `llm.endpoints` / `llm.roles`：各角色的服务与模型。YAML 只保存环境变量名，不保存密钥值。
- `llm.cache`：运行内的分阶段/分片/角色 SQLite 响应检查点；`immutable` 可避免失败恢复时重复调用模型，`none` 可在站点不允许共享存储 SQLite 锁时关闭。
- `extraction.*`、`entity_resolution.llm_max_*`：限制单 chunk 事实数、抽取字段长度、消歧 evidence 和 surface 数，防止异常模型输出放大内存或提示词。
- `reader.*`、`dag.*`、`quality.*`：分片、外排、候选扫描、图宽深度及发布阈值；包括最低有效任务数，空数据集默认不能发布。

生产长链任务可通过 `query_realization.seed_file` 指向 analyst-written JSONL。为避免每个实体 shard 重复调用模型解析 seed，v2 要求自定义 seed 预先包含实体无关的 `task_spec_template`：

```json
{"seed_id":"margin-change","seed_query":"比较同一主体两个期间的指标变化。","task_spec_template":{"objective_type":"change","reasoning_form":"retrieve compatible values for two periods and compute change","required_fact_roles":["new-period value","old-period value"],"time_sensitive":false,"requested_output":"direction and magnitude of change"}}
```

先验证配置：

```bash
mint-synth config validate --config config/production.yaml
```

## 输入长度与合成难度

默认单条 `text` 的校验上限是 `input.max_text_chars: 20000000`，按 Unicode 字符计数。JSONL 还受单行 `input.max_jsonl_line_bytes: 67108864`（64 MiB）限制，因此包含大量多字节字符或 JSON 转义时可能先触发行大小限制；Parquet 不受 JSONL 行限制。这个上限不等于模型上下文长度：正文会按 `chunking.max_chars: 6000` 切块，相邻块默认重叠 400 字符，模型一次只处理一个 chunk 及受控证据集合。

数据分为两类难度：atomic 任务主要考查单个 chunk 内可定位事实的理解与计算；long-horizon 任务默认要求至少 2 个独立 `source_id`，深度门允许真实运算深度 2–4，且会经过确定性回放和独立 auditor。当前内置安全拓扑实际产出深度为 2 或 3，覆盖跨期绝对/百分比变化、三期趋势、四事实的跨期比率变化和三来源分支验证；深度为 1 的跨来源比率、均值和双来源核验会被默认 `min_depth: 2` 过滤。因此默认难度更准确地说是“atomic 易到中等，long-horizon 中等到较难”，而不是任意深度的开放式研究题。实际产出还受语料中可兼容数值事实的密度约束，可通过 `dag.min_depth`、`dag.max_depth`、`dag.min_sources`、`dag.allowed_operations`、公式注册表及 seed 文件调整；系统不会用无意义节点凑深度。仅提高 `max_depth` 不会凭空生成更深拓扑，更深公式需要在执行器中实现并补齐回放测试。每条 long-horizon 成品的 `verification.replay` 会保留实际 `depth`、`source_ids`、`fact_ids` 和运算 `trace`，可以据此分层采样或二次统计难度。

## 规划、提交与恢复

只做元数据扫描、冻结输入清单并创建不可变运行目录。生产中应从与 worker 相同的 SIF 执行规划，避免宿主安装与镜像代码指纹不同：

```bash
apptainer exec /shared/images/mint-synth.sif \
  mint-synth plan --config config/production.yaml
```

检查该目录中的 `plan.json` 与生成配置后，可以提交同一个计划，而不重新扫描输入。提交端会再次比较本地 package、辅助文件及 SIF 指纹；因此宿主上的 `mint-synth` 必须由构建该 SIF 的同一份源码安装，否则提交会被拒绝。这个约束只影响登录节点上的轻量 controller，worker 始终运行冻结的 SIF。

推荐直接编辑仓库中的 `slurm/<stage>.sbatch`：其中的 `#SBATCH` 行管理集群资源，`# MINT_MAX_PARALLEL_TASKS=N` 管理该阶段数组并发。`scripts/submit_slurm.sh` 会为各阶段调用 `sbatch --parsable`、自动补充数组范围和日志路径，并用 `afterok` 串起依赖 DAG：

```bash
export FIN_LLM_API_KEY='...'
bash scripts/submit_slurm.sh config/production.yaml
```

也可以提交已经规划好的运行目录，或先做 dry-run。dry-run 会校验所有静态脚本并生成真实 shard 映射，但不会调用 `sbatch`：

```bash
bash scripts/submit_slurm.sh /shared/outputs/runs/<run_id>
bash scripts/submit_slurm.sh config/production.yaml --dry-run
```

等价的原始 CLI 是：

```bash
mint-synth submit --config config/production.yaml --sbatch-dir slurm
```

超过 `slurm.max_array_tasks` 的阶段会自动拆成多个数组，数组内部通过 `run_dir/slurm/maps/<submission_id>/` 的不可覆盖映射文件把局部数组索引映射到真实 shard，因此支持稀疏恢复且不依赖很大的站点 `MaxArraySize`。每次提交都会在运行清单中记录 sbatch 文件和公共 runner 的 SHA-256。推理服务不由本工具启动或停止。

不传 `--sbatch-dir` 时仍保留旧的动态生成模式；这时资源来自 YAML 的 `slurm.stage_resources`，未配置的阶段使用 4 CPU、16G、4 小时默认值。

```bash
mint-synth status --run-dir /shared/outputs/runs/<run_id>
bash scripts/resume_slurm.sh /shared/outputs/runs/<run_id>
mint-synth validate --run-dir /shared/outputs/runs/<run_id>
```

`resume` 只提交缺失或损坏的 shard。每个 shard 的签名包含运行指纹、输入 assignment、上游签名和引擎版本；数据先写到同一文件系统的临时目录，成功后通过原子 rename 提交 `_SUCCESS.json`。若 marker 与签名吻合但已提交产物缺失或大小异常，旧目录会被移动到该阶段的 `corrupt/` 留证，再按原计划重算；不会静默覆盖或删除。

恢复前会通过 `squeue` 检查历史 job ID；只要 job 仍被 live queue 返回（包括 `REQUEUED` 等站点状态），就拒绝重复提交。只有在人工确认并取消旧作业后，才应使用 `resume --force-resubmit`。模型响应检查点位于 `cache/llm/<stage>/`，因此 shard 失败后可复用已经成功返回的请求。

## 输入语义

JSONL 每个非空行必须是 JSON object；Parquet 每个 row 必须含配置的正文列。记录被规范化为：

```text
record_id, content_hash, text, source_uri, source_format,
source_locator_json, external_id, metadata_json,
dedup_bucket, shuffle_partition, external_id_partition
```

JSONL 用字节范围切分，worker 会对齐换行边界；Parquet 按 row group 切分。整个语料不会读入单机内存。正文完全相同的记录会在调用 LLM 前全局去重；外部 ID 先通过独立 hash shuffle 汇合，所以即使冲突正文原本落在不同内容分区，同一 `external_id` 指向多个正文哈希时仍会被全局发现，相关记录全部进入 quarantine，不会静默择一。

压缩 JSONL 未被当作可切分输入支持。请先转换为普通 JSONL 或 Parquet。

## 运行目录

```text
runs/<run_id>/
  resolved_config.yaml
  input_manifest.parquet
  plan.json
  run_manifest.json
  assignments/normalize/*.json
  cache/llm/<stage>/*.sqlite3
  stages/<stage>/shards/<id>/
    *.parquet
    quarantine.parquet
    _SUCCESS.json
  stages/<stage>/failures/*.json
  slurm/submissions/<submission_id>/{jobs,scripts}/
  slurm/maps/<submission_id>/*.txt
  slurm/logs/*.out|*.err
  logs/<stage>/*.jsonl
  exports/manifest.json
  _PUBLISHED.json
```

Parquet 是规范产物；`output.export_formats` 可额外启用 JSONL。只有所有依赖 shard 完成、有效任务数不少于 `quality.min_validated_tasks` 且全局质量阈值通过后，export stage 才写 `_PUBLISHED.json`。发布清单默认记录每个数据产物及 `quality_summary.json` 的 SHA-256；首次发布时间固化在已提交清单中。若 export shard 已原子提交但根发布标记写入时进程中断，再次执行 export 会幂等补写标记而不改写发布时间。

## 故障与质量门

- 单条脏记录、事实或任务进入带原因和来源定位的 quarantine。
- 输入变更、上游 shard 缺失、外排失败等系统错误会令 shard 失败，并阻止下游 `afterok` 作业。
- `quality.max_invalid_record_ratio`、`max_external_id_conflict_ratio`、`max_llm_terminal_error_ratio` 和 `max_validation_failure_ratio` 会在完整上游阶段聚合后检查，并在全局发布前再次检查。
- auditor 与 constructor 可以配置为不同模型；确定性验证不会信任 auditor 的结论，会再次回放图并检查证据、单位、答案、泄漏和评测集近重合。

## v1 迁移

旧版单 JSONL 参数式流水线保留在 Python 模块中供结果读取，但默认 CLI 已切换到不兼容的 v2。可生成迁移配置：

```bash
mint-synth migrate-v1 --input /path/to/old-input.jsonl --output config/migrated.yaml
```

`--input` 也可指向含旧 `manifest.json` 的输出目录。

## 上线边界

代码具备生产所需的配置冻结、流式 I/O、外排去重、SLURM DAG、幂等 shard、失败隔离、恢复和发布门，但“代码可部署”不等于“已在你的集群验收”。上线前至少需要完成：

1. 用真实 Apptainer、NFS 挂载和推理 endpoint 做 1–10GB 冒烟运行。
2. 做一次节点抢占/超时/429/服务重启故障注入，验证 SQLite 检查点、`resume` 和站点 NFS 锁语义。
3. 用代表性语料标定实体误合并率、事实准确率、DAG 有效率及各质量阈值。
4. 做目标规模的吞吐与 NFS inode/带宽压测，再固定镜像 digest、模型版本和公式注册表版本。

项目是对论文公开方法的工程化复现，不是作者官方实现。论文未公开的 prompt、采样比例、完整 seed bank 和内部 harness 采用了明确、可审计的工程补全；详情见 [methodology.md](docs/methodology.md)。
