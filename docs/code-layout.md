# 代码目录与运行边界

`mint_synth.cli:main` 是命令入口，提供配置校验、计划、本地运行与恢复、状态、worker、发布和迁移命令。业务代码按职责放在以下子包中：

| 子包 | 负责内容 |
| --- | --- |
| `core` | 配置、运行目录与分片状态、指纹、通用函数 |
| `storage` | 输入目录、JSONL/Parquet 读取、记录规范化、切块 |
| `inference` | 模型客户端、角色运行时、提示词 |
| `synthesis` | 事实抽取、实体解析、DAG、题目、审计与去污染 |
| `harness` | 冻结语料快照、可扩展工具注册、会话状态、query 加载与私有答案校验 |
| `sft` | 任务筛选、messages 格式化、教师轨迹 |
| `workflow` | v2/v3 计划、本地运行器、阶段处理器、worker、query 导出和发布 |
| `legacy` | v1 单机流水线兼容实现 |

`workflow/stages` 按处理阶段分成 `ingest`、`facts`、`questions`、`indexes`、`query_export`、`sft_data` 和 `publication`；`registry.py` 保留计划中的阶段名称。query 导出读取已验证任务并生成纯 query 与隐藏校验文件；SFT 数据准备让强教师在 `harness` 中执行 query。query 导出不依赖教师轨迹。项目内提供外部 OPD 可调用的 [Harness API](harness.md)，不包含学生训练或 OPD 轨迹收集模块。

[架构图](project-architecture.html)展示 v3 数据阶段及两种启动方式。本地调用 `mint-synth run`；集群通过仓库根目录的 `scripts/submit_slurm.sh` 提交一个 `slurm/mint-synth.sbatch` 作业，该作业调用同一个 `run` 入口。调度器资源选项不属于合成配置。

包内实现应直接导入所属子包。`mint_synth` 顶层的部分旧模块路径（例如 `mint_synth.dag`、`mint_synth.stages`）保留为兼容入口；原有 `mint_synth.slurm` 控制器已移除。`mint_synth.sft` 从单文件变成同名包，对外仍提供原有的主要函数。

代码指纹递归覆盖 `src/mint_synth` 下的 Python 文件，并覆盖内置 JSON/YAML 资源。v3 自定义工具包及声明的资源文件另计入辅助指纹。目录重构会改变指纹，因此使用旧代码创建的计划仍须由对应版本的代码执行；新计划通过 `run --run-dir` 跳过签名吻合的已完成分片。数据文件格式和导出路径不因运行方式改变。
