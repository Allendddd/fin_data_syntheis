# 单脚本 SLURM 提交

SLURM 只负责启动合成程序。v3 YAML 不含分区、队列、CPU、容器或 Python 路径等调度字段。本地和集群都执行同一个 `mint-synth run --run-dir ...`；作业中断后再次执行会跳过已完成分片。

先在登录节点规划：

```bash
mint-synth plan --config config/production.yaml
```

取得输出中的 `run_dir` 后提交一个作业：

```bash
export FIN_LLM_API_KEY='...'
export MINT_PYTHON="$HOME/.venvs/mint-synth/bin/python"
export MINT_WORKERS=2
bash scripts/submit_slurm.sh /shared/outputs/runs/<run_id> \
  --partition=compute --cpus-per-task=4 --mem=16G --time=12:00:00
```

仓库只有一个 [作业脚本](mint-synth.sbatch) 和一个 [提交脚本](../scripts/submit_slurm.sh)。提交脚本将额外参数原样传给 `sbatch`，所以可以按站点传入 `--account`、`--qos`、`--gres` 等选项；默认资源也可直接编辑作业脚本。`MINT_PYTHON` 应指向计算节点可访问、已安装本项目依赖的 Python；计算节点还需访问同一份代码、输入、工作目录和输出目录。这里直接使用本地 Python 环境，不启动容器，也不需要容器 bind。`MINT_WORKERS` 控制并行分片进程数，默认 1；应结合每个进程的 DuckDB 内存需求选择。

作业退出后，用同一命令和运行目录重新提交即可续跑。运行锁会拒绝两个作业同时处理同一目录；`mint-synth status --run-dir ...` 可查看完成量。SLURM 日志写入 `<run_dir>/logs/slurm-<job_id>.out` 和 `.err`。项目本身无需 SLURM，直接在本机运行的命令见 [README](../README.md)。
