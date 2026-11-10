# mint-synth

教师 SFT 轨迹与项目外 OPD rollout 共用的 harness 用法、状态恢复和自定义工具接口见 [Harness 文档](docs/harness.md)。

在 Linux 上采集 Claude Code/Codex agent 事件与 vLLM/SGLang 模型请求，见 [Agent 轨迹采集](docs/agent-tracing.md)。

长程教师合成可配置 Strands、Codex 或 Claude Code 作为执行 runtime，统一调用金融工具并保留原始轨迹，见 [教师 runtimes](docs/teacher-runtimes.md)。

金融数据合成流水线：从 JSONL/Parquet 语料构造并验证长程 query，再按需让强模型在冻结语料的 harness 中执行 query，采集 SFT `messages` 轨迹。query 可交给项目外的 OPD 流程；本项目提供共用的执行与私有校验环境，不包含模型训练器、学生策略调用或参数更新。

- [数据构造方法](docs/methodology.md)
- [长程 query 与外部 OPD 数据契约](docs/opd.md)
- [SFT 轨迹与输出格式](docs/sft-v3.md)
- [代码目录](docs/code-layout.md)
- [单脚本 SLURM 提交](slurm/README.md)
- [合成流程示意](docs/data-synthesis-flow.html)

## 安装

使用 Python 3.10–3.13。本地运行和 SLURM 作业调用同一个 `mint-synth run` 入口。

```bash
python -m venv ~/.venvs/mint-synth
~/.venvs/mint-synth/bin/python -m pip install -c constraints.txt -e .
source ~/.venvs/mint-synth/bin/activate
```

Windows 本地开发可使用 `.venv\Scripts\python.exe`；SLURM 计算节点需要可执行的 Linux Python。解释器由命令或环境变量选择，不写入数据配置。集群运行时，计算节点需能访问同一份代码、Python 环境、输入、工作目录和输出目录。

## 配置

复制 [示例配置](config/mint-synth.example.yaml)，修改输入、输出和模型服务：

```bash
cp config/mint-synth.example.yaml config/production.yaml
~/.venvs/mint-synth/bin/python -m mint_synth config validate --config config/production.yaml
```

主要字段：

- `input.paths`、`input.text_field`、`input.metadata_fields`：语料和正文、日期、实体等元数据。`input.available_at_field` 需要列入元数据。
- `run.cutoff`、`run.work_dir`、`output.root`：冻结截止日、工作目录和输出目录。
- `llm.endpoints`、`llm.roles`：外部 OpenAI-compatible `/chat/completions` 服务。密钥只通过配置指定的环境变量传入。
- `query_realization.seed_file`：可选的长程任务种子 JSONL。
- `query_export.enabled`：导出经过验证的长程 query；`min_train_queries` 是发布门槛。
- `training.enabled`：是否采集教师 SFT 轨迹；`training.teacher_role` 可选择单独配置的 `llm.roles.teacher`。其余 `training.*` 字段只用于数据筛选、教师 rollout 和 SFT 样本门槛，不执行模型训练。

只合成长程 query 时，设置 `training.enabled: false`、`atomic.enabled: false`，保持 `query_export.enabled: true`。标签与种子筛选见 [SFT 数据文档](docs/sft-v3.md)。

## 本地运行与恢复

一条命令创建计划并在当前机器执行：

```bash
export FIN_LLM_API_KEY='...'
mint-synth run --config config/production.yaml --workers 1
```

`--workers` 控制同时执行的分片进程数。新计划的运行目录会在执行前打印到标准错误输出；中断后用该目录继续，已完成分片会按签名跳过：

```bash
mint-synth run --run-dir /path/to/outputs/runs/<run_id> --workers 4
mint-synth status --run-dir /path/to/outputs/runs/<run_id>
mint-synth validate --run-dir /path/to/outputs/runs/<run_id>
```

## SLURM 提交

先创建同一份与调度器无关的计划，输出中包含 `run_dir`：

```bash
~/.venvs/mint-synth/bin/python -m mint_synth plan --config config/production.yaml
```

在登录节点提交单个作业。`MINT_PYTHON` 指向计算节点可用的 Python；分区、CPU、内存和时限通过 `sbatch` 参数或 [唯一的作业脚本](slurm/mint-synth.sbatch) 设置：

```bash
export FIN_LLM_API_KEY='...'
export MINT_PYTHON="$HOME/.venvs/mint-synth/bin/python"
export MINT_WORKERS=2
bash scripts/submit_slurm.sh /shared/outputs/runs/<run_id> \
  --partition=compute --cpus-per-task=4 --mem=16G --time=12:00:00
```

作业结束或中断后，再提交同一个运行目录即可续跑：

```bash
bash scripts/submit_slurm.sh /shared/outputs/runs/<run_id> --partition=compute
```

分片成功后写入 `_SUCCESS.json`，恢复时按冻结签名复用。运行锁会阻止同一目录被两个 `run` 进程同时执行。输入、代码或辅助文件改变后，已有计划不会直接续跑。SLURM 只负责提交作业，合成配置不含调度参数；详见 [SLURM 说明](slurm/README.md)。

## 产物

已验证的长程任务先独立写入 `stages/query_export/shards/000000/queries/`：

```json
{"query":"比较某公司两个报告期的营业利润率变化，并给出依据。"}
```

`train.jsonl` 和 `val.jsonl` 每行只含 `query`。同目录的 `hidden-verifier.jsonl` 保存任务 ID、答案图、证据及快照索引，仅供外部 harness 校验，不能放入模型输入。教师 SFT 轨迹采集失败时，已完成的 query 分片仍可使用。最终发布后，query 也复制到 `stages/export/shards/000000/queries/`。

启用 SFT 数据准备时，`stages/export/shards/000000/sft/` 中的 `long-train.jsonl`、`long-val.jsonl` 是通过答案与证据验证的教师 harness 轨迹；`atomic-train.jsonl`、`atomic-val.jsonl` 是原子任务样本。它们采用 `messages` 格式，`audit-sidecar.jsonl` 保存不进入模型输入的标签、答案和证据。详见 [SFT 文档](docs/sft-v3.md)。

最终发布目录还含原始审计任务数据、`quality_summary.json`、`artifact_manifest.json`，运行根目录生成 `_PUBLISHED.json`。任务原始数据格式由 `output.export_formats` 控制。query 与 SFT 分片的生成顺序和外部使用边界见 [长程 query 文档](docs/opd.md)。

## 迁移

```bash
mint-synth migrate-v1 --input /path/to/old-input.jsonl --output config/migrated-v2.yaml
mint-synth migrate-v2 --input config/migrated-v2.yaml --output config/production.yaml --cutoff YYYY-MM-DD
```

已冻结的旧计划仍需用其创建时的代码版本执行。项目是对论文公开方法的工程化复现，未公开的 prompt、种子库和内部 harness 使用可审计的工程补全，见 [方法文档](docs/methodology.md)。
