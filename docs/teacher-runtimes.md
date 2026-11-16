# 数据合成的教师 agent runtimes

长程教师轨迹支持 `native`、`strands`、`codex`、`claude_code`。这是离线合成数据功能：模型执行金融 query，合格结果进入训练数据，每次尝试的原始事件与模型调用单独留档。Strands、Codex、Claude Code 均支持 `finance` 和 `native` 两种工具模式：前者使用统一金融工具并导出 ms-swift 数据，后者使用框架原生工具并按实际模型协议逐请求导出。query/DAG 构造、原子教师生成与私有校验继续使用现有流程。

实现参考了 [Slime agent runtime adapters](https://github.com/THUDM/slime/blob/main/docs/en/get_started/agent.md)、[Codex/Claude Code harness](https://github.com/THUDM/slime/tree/main/slime/agent/harness) 和 [Strands 集成示例](https://github.com/THUDM/slime/blob/main/examples/strands_sglang/generate_with_strands.py) 的每会话执行和模型调用采集思路。本项目没有 Slime 运行依赖，也不在这里进行 RL 更新。Slime 的模型适配器另外处理采样 token/logprob；这里的原始 HTTP 轨迹保留服务实际返回的字段，不会从文本推造 token IDs。

## Linux 安装与配置

代码在工作站编写，在 Linux 服务器运行。使用项目支持的 Python 3.10–3.13 环境安装：

```bash
python3 -m pip install -e .
# 使用 Strands 时安装可选依赖：
python3 -m pip install -e '.[strands]'
```

Codex 和 Claude Code 需要在同一 Linux 环境中预先安装 CLI。实现使用 CLI 的非交互参数和独立临时配置；金融模式连接 MCP，原生模式使用各自内置工具。Strands 原生模式使用官方 `strands_tools` 工具包。应固定 SDK、工具包、CLI 与模型服务版本，并在目标服务器做真实模型 smoke test。项目不会自动安装或升级 CLI。

在 v3 配置中指定长程教师的 runtime：

```yaml
training:
  long_teacher_roles: [teacher]
  agent_runtime:
    default: strands       # native / strands / codex / claude_code
    by_role: {}
    commands:
      codex: [/opt/bin/codex]
      claude_code: [/opt/bin/claude]
    timeout_seconds: 600
```

`commands` 是可选的 argv 前缀，默认寻找 PATH 中的 `codex`、`claude`，不经过 shell。可用包装脚本固定安装环境。不同角色可以选不同 runtime：

```yaml
llm:
  roles:
    teacher_strands: {endpoint: inference, model: your-model, temperature: 0.2}
    teacher_codex: {endpoint: inference, model: your-model}
    teacher_cc: {endpoint: inference, model: your-model}
training:
  long_teacher_roles: [teacher_strands, teacher_codex, teacher_cc]
  rollouts_per_task: 3
  agent_runtime:
    by_role:
      teacher_strands: strands
      teacher_codex: codex
      teacher_cc: claude_code
```

其余参数由 schema 补默认值。现有角色轮换仍在首条合格轨迹出现时停止；上例不保证每个任务运行全部三种 runtime。做框架对照时，应分别运行相同任务集，保留每种配置与版本。

模型和凭据复用 `llm.roles.<role>` 与 `llm.endpoints.<endpoint>`。外部 runtime 的 `base_url` 必须是模型服务 origin 或以 `/v1` 结尾，不能含凭据、查询参数或其他路径前缀。

## 三种框架的原生工具模式

`training.agent_runtime.default: native` 表示项目原有 JSON 动作循环；`codex.mode: native`、`strands.mode: native`、`claude_code.mode: native` 才表示所选框架使用自身原生工具。三者的 `mode` 默认均为 `finance`，已有配置保持原行为。

```yaml
training:
  long_teacher_roles: [teacher_codex, teacher_strands, teacher_cc]
  agent_runtime:
    by_role:
      teacher_codex: codex
      teacher_strands: strands
      teacher_cc: claude_code
    codex:
      mode: native
      sandbox: workspace-write
    strands:
      mode: native
      tools: [file_read, shell]
    claude_code:
      mode: native
      tools: [Read, Bash]
      allowed_tools: [Read, "Bash(cat *)", "Bash(rg *)", "Bash(python3 *)"]
    timeout_seconds: 600
```

原生模式为每次尝试准备独立工作目录，把同一实体在截止日内可见的语料分块写成普通文件，保留干扰材料和公开清单。模型通过框架自带的文件、命令等工具自行研究，正常最终回复给出自然语言金融推导和答案。不会启动 `FinanceToolBridge`、金融 `HarnessSession` 或 MCP，也不注册 `search_corpus`、`open_chunk`、`calculate`、`submit_final`。隐藏图只供结束后的离线校验，不进入工具定义或模型输入。

采集器将模型生成的读取调用和随后真实返回按工具调用 ID 绑定，再核对文件位置、实际返回 quote、语料指纹、来源日期及实际最终解答。原生文件研究与金融工具循环因此可共享答案和证据约束，但保留各自真实的工具名、参数、上下文和模型协议，不把内置工具改名成金融工具。金融会话的 `max_tool_steps` / `max_context_chars` 不控制原生循环；执行期限由 `agent_runtime.timeout_seconds` 控制。

### Codex 原生工具

`codex.mode: native` 保留 Codex 自带工具与默认 system，使用独立 HOME 和语料工作目录，不注入 finance MCP 或金融 system。`codex.sandbox` 可选 `workspace-write` 或 `read-only`；默认前者允许在临时工作目录写计算脚本和研究文件，后者适合只读调查。实现保留 sandbox，不使用 `--yolo` 等 bypass 参数。[Codex sandbox 与 CLI 参数](https://learn.chatgpt.com/docs/cli/reference)

Codex 使用 Responses 协议。模型调用的 function/custom 工具、工具返回和最终消息都保留原始关联 ID；自由文本工具输入不能强行当成 JSON 参数。CLI 最后一个成功回合的正常 assistant 回复须与 HTTP 捕获中的最终响应一致，较早草稿和辅助回复不能代替最终答案。允许的读取形式以离线校验器支持的工具及命令为准；未识别的任意脚本输出不会仅因包含正确 quote 就被判为来源读取。

### Strands 官方工具

`strands.mode: native` 加载配置的官方 `strands_tools` 模块，默认 `file_read` 与 `shell`；不会用 `PythonAgentTool` 再包装自定义金融动作。SDK 维护原生 agent loop 和工具消息，模型使用角色配置的 OpenAI Chat Completions endpoint。

当前沿用 SDK 1.x 接口，工具包限定 `strands-agents-tools>=0.8,<0.9`，可选依赖安装会同时安装它；0.9 已迁移旧文件工具，因此不要单独无上限升级。可配置 `file_read`、`shell`、`file_write`、`python_repl`，由工具包提供实际定义和执行实现。[官方工具迁移说明](https://strandsagents.com/docs/user-guide/migrate/strands-agents-tools/)

原生 Strands 在独立子进程和工作目录中运行，避免并发任务通过全局 `chdir` 或环境变量互相影响。文件读取仍须能绑定到工作区内实际来源和真实工具返回；命令工具可用于检索和计算，但不能通过打印一个已知引文来伪造取证。运行的原生消息、事件、最终回复和 HTTP 捕获分别保存，结束后再做离线校验。

## Claude Code 金融模式与原生模式

`training.agent_runtime.claude_code.mode` 默认为 `finance`，保持关闭常规内置工具、自定义金融 system 和统一金融 SFT 导出。切换为原生模式：

```yaml
training:
  long_teacher_roles: [teacher]
  agent_runtime:
    default: claude_code
    claude_code:
      mode: native
      tools: [default]
      allowed_tools: []
```

原生模式只使用 CC 内置工具，保留 CC 默认 system，不传 `--system-prompt`、`--bare` 或 `--disable-slash-commands`。`tools` 控制内置工具集合，`[default]` 使用 CC 默认集合，也可用 `[Read, Bash]` 指定子集，`[]` 关闭常规内置工具。工具集合随 CC 版本变化，例如当前 Linux 默认集合不包含 Grep/Glob，仍可用 Bash 搜索。CC 的原生模式与本项目旧 JSON 动作 runtime `default: native` 是两个不同设置。[CC CLI 工具参数](https://code.claude.com/docs/en/cli-reference)

每次运行将该实体在截止日内可见的全部语料分块写成普通 `corpus/*.txt` 文件，公开的 `corpus-manifest.json` 仅包含来源、日期、chunk ID、路径和指纹；包含干扰材料，不按隐藏答案挑选文件。问题、文件位置和正常最终回复要求作为用户输入提供，隐藏答案图保留在采集器中。CC 自行检索、阅读、运算，直接在正常最终回复中给出自然语言推导与答案。

原生模式不会创建 FinanceToolBridge、HarnessSession 或金融 MCP 服务；传入空 MCP 配置、`--strict-mcp-config` 和 `--disallowedTools "mcp__*"`，采集/导出也拒收 MCP 调用及 MCP 工具定义。`search_corpus`、`open_chunk`、`calculate`、`submit_final` 均不注入。

离线检查器将真实模型响应中的原生 `Read` 调用与后续请求中的 `tool_result` 按 ID 关联，再与冻结文件的指纹、来源、日期和引用核对；也识别明确路径的 cat/head/tail/sed/rg/grep、相对路径及 workdir、受限 Python 文件读取。截断读取只认可实际返回的文本，打印文字、读取其他文件或错误工具结果不能充当证据。最终解答 必须来自 CLI 最后一条主会话回复，并匹配原始模型响应；独立审核实际解答并回放候选自身运算、单位和证据。任意脚本不因打印正确引文就获得来源信用；仅识别能绑定到真实文件、成功调用和实际返回内容的形式，无法绑定时留因。这里回放证据和 proof，不重新执行任意命令或还原整个临时工作目录。

两种模式都使用独立临时 HOME、工作目录和 `dontAsk` 权限模式。`allowed_tools` 仅在原生模式下设置 CC 的自动许可规则，例如 `[Read, "Bash(rg *)", "Bash(cat *)", "Bash(python3 *)"]`；需要执行计算脚本时应配置相应规则。工具可见性与执行许可分开配置。参见 [CC CLI 工具和权限参数](https://code.claude.com/docs/en/cli-reference)。

原生模式的原始 HTTP 请求/响应和 CLI 事件仍完整留档；训练样本写入 `sft/cc-native-train.jsonl`、`sft/cc-native-val.jsonl`，不混入统一金融 `long-*.jsonl`。

## 原生 CC 训练数据契约

原生 CC 每次成功完成的模型请求导出一行 `claude_code_native_v2` 样本。输入来自该次真实请求，新增的最后一条 assistant 消息来自该次真实响应：

```json
{"sample_format":"claude_code_native_v2","trajectory_id":"...","call_id":"...","model":"...","system":[{"type":"text","text":"<实际 CC system>"}],"tools":[{"name":"Read","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"<实际请求上下文>"},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Read","input":{"file_path":"notes.txt"}}]}],"target_message_index":1,"loss_mask":[false,true],"target_block_loss_mask":[true]}
```

示意中的工具 schema 已缩略；实际文件保留该次请求的完整原生 `tools`，包括未调用的工具及 `input_schema`、延迟加载等属性。保留原始 `system` 内容块、历史 `tool_use`/`tool_result`、关联 ID、thinking 文本和签名、工具原名及正常最终回复；不另造 memory assistant 消息或金融工具。工具集合变化时，各请求使用各自的定义，避免一份全局工具清单覆盖真实上下文。

`loss_mask` 对请求历史均为 false，仅新增响应为 true；`target_block_loss_mask` 进一步标记该响应内的监督范围，实际模型文本和原生调用参与监督，redacted thinking 与工具结果块不作为预测目标。CC 原生工具不使用金融答案图的逐动作相关性 mask。`response_metadata` 保留响应 ID、stop reason、usage 等非正文元数据。

这是保留 Anthropic 内容块的协议数据，**不能直接当作 ms-swift 的统一金融 `messages/tool_call` 格式使用**。训练端需按学生模型的模板处理内容块和 loss mask；签名、redacted 数据、usage 不能当作自然语言推理目标。原始模型字节始终保存在 `model/calls`，SSE 会校验结束事件并重建可见内容块。无法重建的事件、未声明的客户端工具、MCP 定义/调用、缺失来源读取或最终证明错误时明确拒绝该轨迹。

同一任务的全部请求使用相同 train/val 划分。`min_long_samples` 按接纳的完整金融轨迹计数，原生模式的多次模型请求不会放大最低轨迹门槛。`manifest.json.formats` 区分各文件格式；native 的 `trajectory.json.sample_origins` 将每行绑定到对应原始请求/响应。

CC 子任务可能并发发起请求，来源读取按原生 `tool_use_id` 绑定真实返回，而不是把请求开始顺序当作执行顺序。`native-audit.json.read_bindings` 及各样本的来源映射记录读取调用、chunk ID 和包含返回的请求位置。最终回复按主会话模型消息 ID 绑定，合并同一 ID 下逐块产生的 assistant 事件；不能用较早草稿、子任务答案或最终回答之后的读取补证据。[CC 内容块事件顺序](https://code.claude.com/docs/en/agent-sdk/streaming-output#message-flow)

## Codex 与 Strands 原生训练数据契约

两种原生模式均按成功捕获的实际模型请求导出，每个请求一行，携带 `sample_format`、`trajectory_id`、`call_id`；上下文、工具定义与模型输出以该次请求为准，不拼成一条伪造的统一金融会话。

| runtime 原生模式 | sample_format | 发布文件 | 原生内容与监督字段 |
| --- | --- | --- | --- |
| Codex | `codex_native_v2` | `codex-native-train.jsonl` / `codex-native-val.jsonl` | 复制实际 Responses request，保留 `input`、`tools`、`instructions`、`previous_response_id` 等；增加实际响应的 `output`、`input_loss_mask`、`target_output_loss_mask`、`response_metadata` |
| Strands | `strands_native_v2` | `strands-native-train.jsonl` / `strands-native-val.jsonl` | 复制实际 Chat Completions request，向 `messages` 追加实际 `choices[0].message`，保留 `tool_calls` 和接口暴露的推理字段；增加 `target_message_index`、`loss_mask`、`response_metadata` |

Codex 的 `input_loss_mask` 将输入列表标为 false，`target_output_loss_mask` 标记本次输出项的监督范围；原生调用按相关性监督，可见推理按关联动作贡献屏蔽；target_reasoning_text_fields 区分可见 content、summary 与不可训练的 encrypted_content。Strands 的 loss_mask 对历史为 false；新增响应同时使用 target_text_loss 和按调用 ID 的 target_tool_call_loss_mask，不能统一设为 true。工具返回位于原始上下文中，不作为本轮 assistant 预测目标。这些 mask 描述样本级监督边界，不能替代学生模板产生的最终 token labels。

Codex 的不透明加密 reasoning 项保留在数据中，但不作为可读推理目标监督。含 `previous_response_id` 的请求保留其原始引用，导出要求被引用响应也在当前采集中；这里不会伪造已展开的完整上下文。

Responses 原生调用可能使用结构化函数参数，也可能使用自由文本 custom tool input；`output` 的 item 类型、call ID、状态与实际字段均保留，不把所有输出压成 `messages` 文本。Codex、Strands 与 CC 的原生文件分别发布，`manifest.json.formats` 标明格式；它们不能直接混入金融 `long-*.jsonl` 或直接当作 ms-swift agent template 输入。训练端需分别适配 Responses items、Chat Completions messages 或 Anthropic 内容块。

同一 task 的全部模型请求使用同一个 split；`sample_count` 表示导出请求数，`min_long_samples` 仍按通过完整离线审核的轨迹数判断。`trajectory.json.sample_origins` 关联每行与原始请求/响应；`native-audit.json` 记录来源读取绑定和最终校验，未完成 HTTP、未声明工具或无法唯一绑定最终回复的尝试不会导出为成功样本。

| runtime | 所需模型协议 | 参数来源 |
| --- | --- | --- |
| `native` | OpenAI Chat Completions，JSON 动作 | 现有角色客户端与缓存 |
| `strands` | OpenAI Chat Completions，原生工具调用 | 角色 temperature/max_tokens；裸 Strands Agent |
| `codex` | OpenAI Responses API | Codex 自身管理请求参数；API trace 保存实际值 |
| `claude_code` | Anthropic Messages API | Claude Code 自身管理请求参数；API trace 保存实际值 |

服务需要配置与模型匹配的 tool parser、chat template，并支持相应协议。Strands SDK 使用 `strands-agents[openai]>=1.35,<2`。Codex 和 Claude Code 的 temperature/max_tokens 不通过不存在的 CLI 参数强行设置，因此跨 runtime 的采样参数未必相同。Claude Code 连接非 Claude 模型属于 Anthropic 未支持的组合，应按所部署的模型与 CLI 版本验证。[Claude Code 网关说明](https://code.claude.com/docs/en/llm-gateway)

## 执行和校验

Strands 金融模式使用 SDK 工具调用和消息历史；Codex 和 Claude Code 金融模式保留各自的 agent loop，通过临时的本地 MCP HTTP 服务调用金融工具。金融模式下每条 rollout 的 `HarnessSession` 独立，工具调用串行进入其拥有者线程，避免 SQLite 跨线程访问和证据状态竞争。三种框架的原生模式独立采集文件研究轨迹，再离线校验。

金融模式的操作经过 `session.step()`，更新预算和证据账本。`submit_final(answer: string)` 转换为环境最终提交；该模式中 SDK/CLI 自然结束的一段文本不能替代提交。私有 `grade()` 与隐藏任务不作为模型工具暴露。可恢复错误留作观察并允许继续；缺证据、解答错误或预算内未提交时拒绝，错误/重复/无贡献动作和相关推理屏蔽 loss。原生模式使用正常最终回复和离线校验；命令或 SDK 返回成功也不等于 内容已通过校验。

默认金融模式使用专门的工具调用提示词和各 runtime 自身的上下文管理；其模型可见历史与 `native` JSON 状态循环不同。统一金融数据从已捕获的模型响应提取可见文本及工具调用，再与环境执行逐项绑定，最终 `submit_final` 格式转换为 assistant 自然语言解答。三种框架的原生模式按各自协议保留请求和响应。SDK/CLI 原始历史单独留档，不与 HTTP 中同一段模型文本重复拼接。

支持 Chat Completions、Responses 和 Anthropic Messages 的 JSON 响应及 SSE：按响应内容块或流式增量保留已暴露文本，检查终止事件和原始体 SHA-256。加密或 redacted reasoning 留在原始响应中，不伪造可读文本。无法解释的内容、缺失/截断的模型响应或与实际工具执行不符的动作，会明确拒绝该样本，原始记录仍保留；不会静默降级成仅含工具的轨迹。当前最终解答 训练协议不接纳提交后的额外文本或与未观察工具调用同批生成的 final。

Claude Code 的默认金融模式禁用常规内置工具并限定 finance MCP；原生模式启用配置的内置工具。Codex 金融模式使用只读 sandbox 并关闭常规命令能力，原生模式启用自带工具及配置的 sandbox。原生 Strands 在子进程内加载官方工具，工作目录与环境独立。默认金融模式的事件审计拒收非金融工具执行，原生模式拒绝 MCP 和注入的金融工具。配置限制不能替代操作系统隔离，部署需要容器时，应在可互通的网络命名空间内运行 runtime 与本地代理。实现没有使用 CLI bypass flags。

## 轨迹产物

每次长程教师尝试写入 `sft_prepare` 分片的 `traces/<attempt_hash>/`，最终发布到 `sft/traces/<shard>/<attempt_hash>/`。成功、失败与超时的尝试均进入 `sft/trajectory-index.jsonl`；`audit-sidecar.jsonl` 中的候选引用也指向发布后的目录。

| 文件 | 内容 |
| --- | --- |
| `result.json` | runtime、模型、状态、是否合格、拒绝原因、原始文件引用 |
| `environment.json` | 环境工具轨迹；结束后的私有校验结果，供审计 |
| `agent.events.jsonl` / `agent.messages.json` | Strands SDK 原始事件与消息历史 |
| `stdout.jsonl` / `stderr.log` / `cli.json` | Codex JSONL 或 CC stream-json、诊断及进程状态 |
| `model/run.json`、`model/calls/*` | 外部 runtime 的请求、响应、SSE 字节、时间和脱敏 HTTP 元数据 |
| `model/calls/000001.json` 等 | native 的实际请求、完整响应 JSON、解析动作、可见内容及缓存状态；完整响应缓存独立于旧的 parsed-only 缓存 |
| `model.parsed.json` | native 的 system/user 输入和解析动作兼容摘要；完整内容见对应的 `source` |
| `trajectory.json` | 完整可导出模型内容、`trajectory_id` 及每条训练消息到模型原始文件/字段/环境步骤的映射 |
| `cc-native.jsonl` | CC 原生模式的逐请求样本；`trajectory.json.sample_origins` 提供来源绑定 |
| `codex-native.jsonl` / `strands-native.jsonl` | Codex / Strands 原生模式的逐请求样本，保留各自实际请求与模型输出 |
| `native-workspace.json` / `native-audit.json` | 原生文件清单、运行结束时的完整性检查、真实读取绑定及离线 proof 校验 |

外部 runtime 自动启动逐次模型调用采集代理，无需手工再开 `trace-proxy`。这与 CLI/SDK 事件互补：API 记录保留实际模型协议和返回内容，框架记录解释执行过程，环境记录证明工具确实取得了证据。

发布的 `long-train.jsonl` / `long-val.jsonl` 包含模型实际返回的分析/规划、可选 memory 笔记、工具交互、最终解答、工具定义及非模型输入元数据 `trajectory_id`；隐藏校验与原始轨迹保存在审计产物中。native 原始记录不写认证头；外部 HTTP 元数据对凭据头脱敏。Linux 上轨迹目录及文件分别使用 `0700`/`0600`；原始 prompt、模型输出和来源正文没有通用脱敏。

超时会保留已完成的金融状态转移和已收到的 API 字节，并标记为失败/截断。未完成的 HTTP 流可能没有完整响应，不能按成功响应回放。框架超时会取消 Strands 或终止 Linux CLI 进程组；自定义 Python 工具本身若阻塞，仍需在工具实现中设置有界执行时间。

## 验证范围

自动测试用于核对完整响应缓存、分析/工作笔记导出、三种协议的响应及 SSE、原生模式配置与执行参数、工具定义和调用 ID、逐请求上下文/监督范围、金融模式回归、发布引用与完整轨迹门槛。模型响应使用 mock，Strands 使用 SDK stub，CLI 使用 fake CLI；本机没有运行 Linux GPU 服务或真实 CLI 的端到端调用，也未用真实学生模型的训练模板核对 token labels。目标服务器应先验证所选模式的原生工具、采集、训练模板与最终证明，再扩大合成任务量。

## 自然语言审核与 v2 迁移

所有长程 SFT 路径使用私有 training.long_verifier_role（默认 auditor），在真实捕获冻结后审核实际最终回答、可见 CoT、取得的来源及动作贡献，额外增加一次独立内容审核。数值与引文可本地核验，语义解释和提取仍依赖审核模型，不宣称完全确定性。

原生 v2 mask 必须由训练适配器消费；输入历史/观察不训练，opaque/redacted 字节不作为推理目标。可见摘要保留其来源，不冒充隐藏 CoT。新代码、prompt 和格式使用新运行目录，不能绕过签名续跑旧计划。真实 GPU、SDK/CLI 和学生模板 labels 仍需目标环境验证。
