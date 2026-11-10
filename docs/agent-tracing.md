# Claude Code / Codex 轨迹采集（Linux 运行）

这套采集分两层：`trace-run` 保存 agent CLI 的 JSONL 事件（消息、工具调用及工具结果）；`trace-proxy` 位于 agent 与 vLLM/SGLang 之间，保存每次模型 API 请求、完整响应和流式片段的接收时间。两者用同一个 `run_id` 和目录。模型服务自身的日志可再提供推理侧请求 ID、token 和性能信息。服务日志单独使用时，看不到 CLI 在本地执行工具后的完整观察结果。

以下命令在 **Linux 服务器**运行。Windows 工作站只用于编写代码。本项目要求 Python 3.10–3.13；在服务器仓库根目录安装：

```bash
python3 -m pip install -e .
```

## 启动本地采集代理

先启动 vLLM 或 SGLang，再在另一个终端启动代理。`--upstream` 填模型服务的 HTTP(S) **origin**，不要加 `/v1`；代理只监听本机 `127.0.0.1`。每个 agent 任务使用独立的目录、`run_id` 和端口。

```bash
umask 077
export RUN_ID='query-000042'
export TRACE_DIR="/data/agent-traces/$RUN_ID"
mint-synth trace-proxy \
  --upstream http://127.0.0.1:8000 \
  --output-dir "$TRACE_DIR" \
  --run-id "$RUN_ID" \
  --port 8765
```

若 agent 和模型服务在不同机器上，代理仍应与 agent 运行在同一个 Linux 网络命名空间，`--upstream` 改为模型服务可访问的地址。API 凭据原样透传；代理不会替你鉴权或转换协议。Claude Code 需要上游支持 Anthropic Messages API（`/v1/messages`）；Codex 需要 Responses API（`/v1/responses`）。请先用部署版本和所选模型确认这两个接口、流式输出及工具调用均可用。[vLLM 的 Claude Code 接口](https://docs.vllm.ai/en/latest/serving/integrations/claude_code/)、[vLLM Responses API](https://docs.vllm.ai/en/latest/serving/online_serving/)、[SGLang Anthropic 接口](https://github.com/sgl-project/sglang/blob/main/docs/docs/basic_usage/anthropic_api.mdx)、[SGLang 接口实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)分别说明了对应路由。Claude Code 对非 Claude 模型的网关路由不在 Anthropic 的支持范围内，因此还需用实际任务验证 CLI 与模型的兼容性。[Anthropic 网关说明](https://code.claude.com/docs/en/llm-gateway)

## 采集 Claude Code

在 agent 所在终端设置环境变量。`SERVED_MODEL` 填服务端的模型名称；若服务端配置了 API key，把占位凭据换成对应的真实值，并避免写入命令行参数。

```bash
export RUN_ID='query-000042'
export TRACE_DIR="/data/agent-traces/$RUN_ID"
export SERVED_MODEL='your-served-model-name'
export ANTHROPIC_BASE_URL='http://127.0.0.1:8765'
export ANTHROPIC_API_KEY='dummy'
export ANTHROPIC_AUTH_TOKEN='dummy'
export ANTHROPIC_DEFAULT_OPUS_MODEL="$SERVED_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$SERVED_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$SERVED_MODEL"

mint-synth trace-run --output-dir "$TRACE_DIR" --run-id "$RUN_ID" -- \
  claude -p '在目标仓库完成你的任务' \
  --output-format stream-json --verbose --include-partial-messages --forward-subagent-text
```

`--forward-subagent-text` 让子 agent 的文本消息也进入 JSONL；工具事件和 `parent_tool_use_id` 可重建调用关系。具体字段以当前 Claude Code 版本为准。[Claude Code 非交互与流式输出文档](https://code.claude.com/docs/en/headless)

## 采集 Codex

Codex 自定义 provider 必须连到支持 Responses API 的服务。下面用一次性 `-c` 参数配置；`supports_websockets=false` 使模型调用走此代理支持的 HTTP/SSE 通道。若部署服务使用 API key，将 `LOCAL_MODEL_API_KEY` 设置成真实凭据。

```bash
export RUN_ID='query-000043'
export TRACE_DIR="/data/agent-traces/$RUN_ID"
export LOCAL_MODEL_API_KEY='dummy'

# 先用这个 RUN_ID 和 TRACE_DIR 启动代理；若上一个代理仍在运行，请改用空闲端口并同步修改 base_url。
mint-synth trace-run --output-dir "$TRACE_DIR" --run-id "$RUN_ID" -- \
  codex exec --json \
  -c 'model_provider="trace"' \
  -c 'model_providers.trace.name="Local trace"' \
  -c 'model_providers.trace.base_url="http://127.0.0.1:8765/v1"' \
  -c 'model_providers.trace.wire_api="responses"' \
  -c 'model_providers.trace.env_key="LOCAL_MODEL_API_KEY"' \
  -c 'model_providers.trace.requires_openai_auth=false' \
  -c 'model_providers.trace.supports_websockets=false' \
  -m 'your-served-model-name' \
  '在目标仓库完成你的任务'
```

`codex exec --json` 输出 agent 事件 JSONL。自定义 provider 的 `base_url` 和 `wire_api` 配置见 [Codex 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)；Responses 路由需求见 [网关兼容要求](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility)。若 CLI 使用额外的远程服务或 WebSocket，此代理无法覆盖那些连接。

## 输出与关联

| 文件 | 内容 |
| --- | --- |
| `run.json` | `run_id`、上游地址 |
| `agent.json` | agent 命令、退出状态、原始事件文件名 |
| `agent.stdout.jsonl` | CLI 原始 JSONL；Claude Code/Codex 的工具事件在这里 |
| `agent.stderr.log` | CLI 诊断输出 |
| `agent.timing.jsonl` | 每行 CLI 输出的 UTC 接收时间、流名、原文件字节偏移 |
| `calls/<call_id>.json` | 每次模型请求的时间、HTTP 状态、脱敏头、SHA-256、完成状态 |
| `calls/<call_id>.request.bin` | 发往模型服务的原始请求体 |
| `calls/<call_id>.response.bin` | 模型服务返回的完整响应体；SSE 也原样保存 |
| `calls/<call_id>.chunks.jsonl` | 响应片段的 UTC 接收时间、相对时间与字节偏移 |

所有调用都带 `run_id` 和唯一 `call_id`。代理给上游请求补 `X-Request-Id: <call_id>`（如果 CLI 未提供），并在下游响应加 `X-Mint-Trace-Call-Id`。服务端若记录或回传 `X-Request-Id`，可据此把服务日志和代理调用对齐。CLI 事件与调用可按 UTC 时间排列；并发子 agent 的精确归属还需 agent 事件中的父调用 ID，或在发起模型请求时传入可关联的请求 ID。`agent.timing.jsonl` 的时间是采集器收到整行事件的时间，不能当作工具执行的精确起点。

项目内的 `HarnessSession.audit_record()["trace"]` 另含 harness 工具动作、观察与证据计数。若 agent 实际通过该 harness 执行工具，可令 `run_id` 等于 query ID，把审计记录与这里的 CLI/模型轨迹按任务关联。未经 harness 执行的 Claude Code/Codex 原生工具调用不会自动进入 `HarnessSession.trace`。私有 verifier 及答案不应进入发给模型的请求体。

轨迹含完整 prompt、模型输出和可能的敏感数据。代理在 Linux 上把目录设为 `0700`、文件设为 `0600`，并在 JSON 元数据中遮盖常见凭据头及查询参数；原始请求体和响应体未做内容脱敏。采集目录应放在受控存储中，按数据保留策略处理。

## 模型服务原生日志

- **vLLM**：`--enable-log-requests --enable-log-outputs --enable-request-id-headers`。INFO 级别主要记录请求 ID、参数和输出文本；若需要输入 prompt，可按部署版本设置 DEBUG 日志。详细 OpenTelemetry trace 面向推理性能分析。[vLLM serve 参数](https://docs.vllm.ai/en/stable/cli/serve/)
- **SGLang**：`--log-requests --log-requests-level 3 --log-requests-format json --log-requests-target /secure/sglang-requests` 可记录完整输入输出；`--enable-trace --otlp-traces-endpoint ...` 用于 OpenTelemetry 性能链路。[SGLang server 参数](https://github.com/sgl-project/sgl-project.github.io/blob/main/markdown/advanced_features/server_arguments.md)

原生日志有助于检查模型侧 token、排队和生成行为。这里的代理负责保留 API 实际收发内容和完整 SSE，CLI JSONL 负责 agent 的消息与工具事件。代理目前支持 HTTP/1.1 请求和 SSE 响应，不代理 WebSocket。
