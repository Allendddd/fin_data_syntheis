# 长程任务 Harness：教师轨迹与外部 OPD 共用环境

本项目的 harness 是有状态的执行环境，可用于强模型的 SFT 轨迹采集，也可由项目外的 OPD 训练器驱动学生 rollout。它负责冻结语料工具、动作门控、上下文装配、证据账本、状态恢复和私有校验；学生策略、关键步骤选择、优势计算和参数更新不在本项目中。

设计依据是 [Mint-Agent 论文第 4 节与第 5.2 节](https://arxiv.org/html/2608.16386)：模型只看到 query、允许的动作、预算、裁剪后的工作状态和工具观察；隐藏答案图与来源包由 harness 的校验端保管。教师先在环境里执行 query，合格轨迹才进入 SFT；OPD 从失败轨迹的动作前状态继续采样。这里实现公开方法的工程契约，不声称复制论文未公开的内部提示词或评判器。

## 数据边界与入口

`query_export` 产出 `train.jsonl`、`val.jsonl`、`hidden-verifier.jsonl` 和 `manifest.json`。前两者每行只有 `query`；隐藏文件记录答案图、证据和快照路径，仅供训练器的环境进程读取。`manifest.json` 记录三份文件、每个冻结 SQLite 快照的 SHA-256，以及工具集指纹。`iter_cases` 校验文件、行绑定、快照路径和校验契约，再产生 `HarnessCase`。模型适配器只接收 `session.observe()`、`session.system_prompt`、`session.tool_definitions` 和每次 `session.step()` 的观察。

```python
from mint_synth.harness import HarnessSession, iter_cases, load_tool_registry

case = next(iter_cases("/path/to/outputs/runs/<run_id>", split="train"))
registry = load_tool_registry()  # 配置了自定义工具时填入 package.module:register_tools

with HarnessSession(case, registry=registry, max_steps=17, max_tool_calls=16) as session:
    visible = session.observe()
    pre_action = session.checkpoint()
    result = session.step({"tool": "search_corpus", "arguments": {"query": "营业收入"}})
    visible = session.observe()
    # 后续由调用方提供动作；结束动作是 {"final": "..."}。
    # episode 结束后，环境进程才调用 session.grade() 取得私有校验结果。
```

`case.task` 和 `grade()` 的返回值包含隐藏答案信息。外部训练器应将它们留在环境进程，不能拼进模型提示词、工具输出或下一步 `observe()`。会话动作每步只能有 `tool` + `arguments`，或 `final`。工具动作可附带 `memory`，包含 `plan`、`confirmed`、`unresolved`；这是模型草稿，不会增加校验用证据。`step()` 不返回隐藏分数。动作格式、工具 JSON Schema、工具调用数和总步数在执行前受到门控；失败观察保留在 `trace` 和 `failed_paths`，不进入 `evidence_ledger`。到预算上限或最终提交时 episode 结束。

会话保存完整 `trace`、`evidence_ledger`、`working_memory`，但 `observe()` 会先裁掉较早的工具历史，再裁证据摘要，使模型视图不超过 `max_context_chars`。来源记录包含快照 chunk ID、来源 ID、可用日期、工具名、获取步和 `artifact_ref`；记录只有在快照中可回查且未越过截止日才会记账。计算记录与来源记录区分，不能单独满足来源证据要求。检索失败或无效动作仍属于完整轨迹，便于外部系统挑选关键步骤。

`checkpoint()` 返回不含隐藏任务的 JSON 可序列化动作前状态。已结束的轨迹可用 `checkpoint_at(step)` 重建第 `step` 次动作之前的状态，用于外部系统选定失败轨迹的关键步骤。`restore(checkpoint)` 会核对 query、快照 SHA-256、工具集指纹、截止日和预算，然后在新会话里重放历史并逐条比较观察。`fork(checkpoint)` 创建隔离分支，供同一关键步骤多次采样。自定义工具必须确定性且只读，否则重放会拒绝漂移结果。checkpoint 应保存在训练器自己的目录；本项目不生产学生轨迹数据集。

`grade()` 只能在 episode 结束后调用，返回 `answer_correct`、`evidence_replayable`、`action_valid`、`cutoff_valid`、`accepted` 和二值 `reward`；`audit_record()` 合并完整轨迹、账本、失败路径与私有判定，供环境进程留档。答案按隐藏 DAG 回放校验；证据路径要求取得每个必需 chunk，且其来源 ID、原文引句和截止日可在冻结快照及实际工具观察中重放；无效动作使整条轨迹不能被接纳。这对应论文中终局答案、证据路径和动作约束三项校验的可确定部分。教师 SFT 采集也调用同一会话与 `grade()`；无关但有效的工具调用可由 `step_relevance` 屏蔽损失。

## 扩展工具

写一个可导入的 Python 模块，并提供接收 `ToolRegistry` 的函数。每个 `ToolSpec` 绑定名称、版本、模型可见的 JSON Schema、执行函数、可选的 SFT 相关性函数与证据提取函数。执行函数只拿到 `SnapshotTools` 和已校验的参数，不拿隐藏任务。下例增加一个沿用快照检索门控的打开工具：

```python
from mint_synth.harness import ToolSpec


def register_tools(registry):
    registry.register(ToolSpec(
        name="open_financial_chunk",
        version="1",
        definition={
            "type": "function",
            "function": {
                "name": "open_financial_chunk",
                "description": "Read a chunk previously returned by search_corpus.",
                "parameters": {
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string"}},
                    "required": ["chunk_id"],
                    "additionalProperties": False,
                },
            },
        },
        handler=lambda tools, args: tools.call("open_chunk", args),
        relevance=lambda task, args, obs: any(
            args["chunk_id"] == fact["chunk_id"] for fact in task["evidence"]
        ),
        distill=lambda tools, args, obs: [{
            "kind": "source_chunk",
            "chunk_id": obs["chunk_id"],
            "source_id": obs["source_id"],
            "available_at": obs["available_at"],
            "excerpt": obs["text"][:300],
        }],
    ))
```

在 v3 配置中声明工厂，教师轨迹采集就会加载同一工具集：

```yaml
harness:
  tool_registry: my_fin_tools.harness:register_tools
  resource_paths: []  # 工具额外读取的本地文件或目录
```

模块须可从本地和 SLURM 使用的 Python 环境导入。规划时会将工具模块所在包的 Python 文件及声明的 `resource_paths` 纳入运行指纹；工具定义或实现变更需要新运行。外部 OPD 调用 `load_tool_registry("my_fin_tools.harness:register_tools")`，会话默认对照 query manifest 的工具集指纹。若要在既有 query 上试验另一套工具，可显式设置 `HarnessSession(..., allow_toolset_override=True)`；审计记录与 checkpoint 仍写入实际使用的工具集指纹，结果须按工具集分别比较。新增工具若返回 `source_chunk`，其 `chunk_id`、`source_id`、`available_at` 和 `excerpt` 必须能在任务实体的冻结快照中核对；不能由模型自己宣称已取得证据。仅返回搜索候选的工具可用 `distill=None`，这些候选不算已获取的来源证据。

自定义工具应只访问冻结、截止日前的数据，并返回 JSON 对象。外部网络、当前时间、随机结果或写入副作用会破坏 checkpoint 重放与 OPD 分支的一致性；需要新数据源时，先把它纳入可指纹化的快照和 query 证据契约，再扩展工具与校验规则。
