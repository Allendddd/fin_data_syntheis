# 数据合成的教师 agent runtimes

长程教师轨迹支持 `native`、`strands`、`codex`、`claude_code`。这是离线合成数据功能：模型在冻结金融语料中执行 query，合格结果进入 SFT `messages`，每次尝试的原始事件与模型调用单独留档。query/DAG 构造、原子教师生成与私有校验继续使用现有流程。

实现参考了 [Slime agent runtime adapters](https://github.com/THUDM/slime/blob/main/docs/en/get_started/agent.md)、[Codex/Claude Code harness](https://github.com/THUDM/slime/tree/main/slime/agent/harness) 和 [Strands 集成示例](https://github.com/THUDM/slime/blob/main/examples/strands_sglang/generate_with_strands.py) 的每会话执行和模型调用采集思路。本项目没有 Slime 运行依赖，也不在这里进行 RL 更新。Slime 的模型适配器另外处理采样 token/logprob；这里的原始 HTTP 轨迹保留服务实际返回的字段，不会从文本推造 token IDs。

## Linux 安装与配置

代码在工作站编写，在 Linux 服务器运行。使用项目支持的 Python 3.10–3.13 环境安装：

```bash
python3 -m pip install -e .
# 使用 Strands 时安装可选依赖：
python3 -m pip install -e '.[strands]'
```

Codex 和 Claude Code 需要在同一 Linux 环境中预先安装 CLI。实现使用当前 CLI 的非交互参数、独立临时配置和 MCP HTTP 工具连接；应固定 CLI 版本，并在目标服务器做真实模型 smoke test。项目不会自动安装或升级 CLI。

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

| runtime | 所需模型协议 | 参数来源 |
| --- | --- | --- |
| `native` | OpenAI Chat Completions，JSON 动作 | 现有角色客户端与缓存 |
| `strands` | OpenAI Chat Completions，原生工具调用 | 角色 temperature/max_tokens；裸 Strands Agent |
| `codex` | OpenAI Responses API | Codex 自身管理请求参数；API trace 保存实际值 |
| `claude_code` | Anthropic Messages API | Claude Code 自身管理请求参数；API trace 保存实际值 |

服务需要配置与模型匹配的 tool parser、chat template，并支持相应协议。Strands SDK 使用 `strands-agents[openai]>=1.35,<2`。Codex 和 Claude Code 的 temperature/max_tokens 不通过不存在的 CLI 参数强行设置，因此跨 runtime 的采样参数未必相同。Claude Code 连接非 Claude 模型属于 Anthropic 未支持的组合，应按所部署的模型与 CLI 版本验证。[Claude Code 网关说明](https://code.claude.com/docs/en/llm-gateway)

## 执行和校验

Strands 使用原生工具调用和 SDK 消息历史；Codex/Claude Code 保留各自的 agent loop，通过临时的本地 MCP HTTP 服务调用金融工具。每条 rollout 的 `HarnessSession` 独立，工具调用串行进入其拥有者线程，避免 SQLite 跨线程访问和证据状态竞争。

所有实际金融操作都经过 `session.step()`，更新预算和证据账本。`submit_final(proof)` 转换为环境最终提交；SDK/CLI 自然结束的一段文本不能替代提交。私有 `grade()` 与隐藏任务不作为模型工具暴露。验收标准保持严格：无效动作、缺证据、错误证明或预算内未提交都会被拒绝。

外部 runtime 使用专门的工具调用提示词和自身上下文管理；其模型可见历史与旧 `native` JSON 状态循环不同。没有声称迁移会自动提升合格率。公共数据输出把实际执行的金融动作与观察规范化成项目现有 `tool_call/tool_response` 格式；该规范化样本不冒充 SDK/CLI 的逐字对话。

Claude Code 配置禁用内置工具，限定 finance MCP 并关闭自动发现。Codex 使用独立临时 HOME、空工作目录、只读 sandbox，关闭已知命令/网络/子 agent 能力；事件审计会拒收非金融工具执行。CLI 配置限制不能替代操作系统隔离，部署需要容器时，应在同一网络命名空间内运行 CLI 与本地代理。实现没有照搬 Slime 假定外层 sandbox 后使用的 bypass flags。

## 轨迹产物

每次长程教师尝试写入 `sft_prepare` 分片的 `traces/<attempt_hash>/`，最终发布到 `sft/traces/<shard>/<attempt_hash>/`。成功、失败与超时的尝试均进入 `sft/trajectory-index.jsonl`；`audit-sidecar.jsonl` 中的候选引用也指向发布后的目录。

| 文件 | 内容 |
| --- | --- |
| `result.json` | runtime、模型、状态、是否合格、拒绝原因、原始文件引用 |
| `environment.json` | 环境工具轨迹；结束后的私有校验结果，供审计 |
| `agent.events.jsonl` / `agent.messages.json` | Strands SDK 原始事件与消息历史 |
| `stdout.jsonl` / `stderr.log` / `cli.json` | Codex JSONL 或 CC stream-json、诊断及进程状态 |
| `model/run.json`、`model/calls/*` | 外部 runtime 的请求、响应、SSE 字节、时间和脱敏 HTTP 元数据 |
| `model.parsed.json` | native 的实际 system/user 输入和解析后 JSON 输出；可来自现有缓存 |

外部 runtime 自动启动逐次模型调用采集代理，无需手工再开 `trace-proxy`。这与 CLI/SDK 事件互补：API 记录保留实际模型协议和返回内容，框架记录解释执行过程，环境记录证明工具确实取得了证据。

发布的 `long-train.jsonl` / `long-val.jsonl` 只包含规范化合成样本和工具定义；隐藏校验与原始轨迹保存在审计产物中。轨迹含完整 prompt、模型输出和来源内容。Linux 上轨迹目录及文件分别使用 `0700`/`0600`，凭据头在 HTTP 元数据中脱敏；原始文本内容没有通用脱敏。

超时会保留已完成的金融状态转移和已收到的 API 字节，并标记为失败/截断。未完成的 HTTP 流可能没有完整响应，不能按成功响应回放。框架超时会取消 Strands 或终止 Linux CLI 进程组；自定义 Python 工具本身若阻塞，仍需在工具实现中设置有界执行时间。

## 验证范围

自动测试覆盖共享金融校验、模型 HTTP 捕获、临时配置隔离、真实子进程与 MCP 通信、超时、非金融工具拒收和发布引用。Strands 使用 SDK stub，CLI 使用 fake CLI；本机没有运行 Linux GPU 服务或真实 CLI 的端到端调用。目标服务器应先验证每种协议的工具调用与最终提交，再扩大合成任务量。
