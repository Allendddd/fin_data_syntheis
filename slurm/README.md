# SLURM 作业模板

每个流水线阶段对应一个可直接审查和修改的 `<stage>.sbatch`。通常只需要按站点修改这些行：

- `#SBATCH --partition`、`--account`、`--qos`
- `#SBATCH --cpus-per-task`、`--mem`、`--time`
- `# MINT_MAX_PARALLEL_TASKS=N`（job array 同时运行的最大 task 数）

不要在模板里固定 `--array`、`--output`、`--error` 或 `--dependency`。提交 controller 会根据冻结的运行计划，通过 `sbatch` 命令行注入这些值，并用 `afterok` 构建阶段依赖。

从仓库根目录提交：

```bash
bash scripts/submit_slurm.sh config/production.yaml --dry-run
bash scripts/submit_slurm.sh config/production.yaml
bash scripts/resume_slurm.sh /shared/outputs/runs/<run_id>
```

每次提交都会将实际使用的模板和 `scripts/run_stage.sh` 冻结到运行目录的 `slurm/submissions/<submission_id>/`，并将 SHA-256 写入 `run_manifest.json`。模板中不要写 API key；密钥应在调用提交脚本前导出为配置指定的环境变量。
