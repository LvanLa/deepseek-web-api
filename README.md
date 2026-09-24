# deepseek-web-api

[![CI](https://github.com/kittors/deepseek-web-api/actions/workflows/ci.yml/badge.svg)](https://github.com/kittors/deepseek-web-api/actions/workflows/ci.yml)
[![Node.js](https://img.shields.io/badge/Node.js-%3E%3D20-339933?logo=node.js&logoColor=white)](package.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

将已登录的 `chat.deepseek.com` Web 会话包装为本地 **OpenAI-compatible API**，支持 Responses API、Chat Completions、流式 reasoning/output、多轮 session、Chrome/CDP 登录态和 prompt-based tool calling。

> An unofficial local OpenAI-compatible API for authenticated DeepSeek Web sessions. Text-only, GitHub-first, and designed for Pi and other OpenAI-compatible clients.

> [!WARNING]
> **非官方项目。** 本项目调用 DeepSeek Web 私有接口，不隶属于 DeepSeek 或 OpenAI。Web API、登录、PoW、风控和前端 worker 可能随时变化，并可能带来账号限制风险。默认只监听 `127.0.0.1`；不要把服务直接暴露到公网。使用前请自行确认服务条款、适用法律和账号风险。

## 功能

- `POST /v1/responses`：流式/非流式 Responses 兼容层
- `POST /v1/chat/completions`：流式/非流式 Chat Completions 兼容层
- `GET /v1/models`、`GET /health`
- 自动连接 CDP 或启动专用 Chrome/Chromium profile
- 复用并刷新 `userToken` + cookies，持久化 owner-only `auth.json`
- DeepSeek Web PoW challenge/worker 求解
- `THINK` → Responses reasoning / Chat `reasoning_content`
- `RESPONSE` → Responses output text / Chat content
- API key 保护全部 `/v1/*` 路由
- DeepSeek session 与 `parent_message_id` 持久化；保存完整展开历史并按折叠指纹匹配，切换 flash/pro 或 thinking、工具轮之后都不主动分叉
- 客户端用滑动窗口回放（只发最近若干轮、丢弃更早的轮次）导致历史匹配失败时，按请求指令的哈希粘回该指令下最近的 session：同一次 agent 会话始终落在同一个 DeepSeek 对话里；全新任务指令文本不同，不会误粘
- system/developer/AGENTS/skills、工具 schema、assistant/tool 历史的 prompt 兼容
- 文本工具协议 → Chat `tool_calls` / Responses `function_call`；容忍真实模型输出的松散标签、DSML 包装和转义错误的 Windows 路径
- Pi `openai-completions` 与 `openai-responses` 配置示例

## 前置要求

- Node.js 20 或更高版本
- pnpm 10
- Google Chrome 或 Chromium
- 能访问并登录 `https://chat.deepseek.com`

## 快速开始

```bash
git clone https://github.com/kittors/deepseek-web-api.git
cd deepseek-web-api
pnpm install --frozen-lockfile
pnpm build
pnpm start
```

默认地址：

```text
http://127.0.0.1:8787
```

首次启动流程：

1. 尝试验证 `data/auth.json`，有效时不打开浏览器。
2. 否则连接 `DS_CDP`（默认 `http://127.0.0.1:9333`）。
3. 默认本地 CDP 不可用时，启动 `data/chrome-profile/` 专用 Chrome。
4. 如未登录，打开可见窗口并等待你完成 DeepSeek 登录。
5. 登录验证成功后原子写入 `data/auth.json`，再启动 HTTP 服务。
6. 日常 PoW/browser 工作默认 headless；`DS_SHOW_BROWSER=1` 可显示窗口。

也可先显式完成登录：

```bash
pnpm login
```

首次读取或生成本地 API key：

```bash
cat data/.api-key
```

## API 示例

```bash
API_KEY="$(cat data/.api-key)"
```

### Responses API

```bash
curl http://127.0.0.1:8787/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "input": "用三句话解释 TypeScript",
    "reasoning": {"effort": "medium"}
  }'
```

流式：

```bash
curl -N http://127.0.0.1:8787/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-pro",
    "input": "证明根号 2 是无理数",
    "stream": true
  }'
```

Responses `output` 可包含 `reasoning`、assistant `message` 和 `function_call`。事件顺序、`previous_response_id` 与支持边界见 [docs/responses-api.md](docs/responses-api.md)。

### Chat Completions

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "x-api-key: $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system", "content": "回答要简洁。"},
      {"role": "user", "content": "你好"}
    ]
  }'
```

普通回答在 `choices[0].message.content`；thinking 在兼容字段 `reasoning_content`。完整字段与限制见 [docs/api.md](docs/api.md)。

## 模型与功能映射

| 公共模型 ID | DeepSeek Web `model_type` | Search |
| --- | --- | --- |
| `deepseek-v4-flash` | `default` | 支持 |
| `deepseek-v4-pro` | `expert` | Web 端不支持，服务端强制关闭 |

兼容别名：

- flash：`flash`、`default`、`deepseek-chat`
- pro：`pro`、`expert`、`deepseek-reasoner`

DeepSeek Web 只提供 `thinking_enabled: boolean`。`none`/`off`/`0`/`false`/`disabled` 映射为关闭；其他 `reasoning.effort` 或 thinking level 都映射为开启。

Web 搜索可通过 `search_enabled: true`、`web_search: true` 或包含 `web`/`search` 类型的工具开启，仅 flash 生效。

## Pi 集成

推荐先使用 Pi 的 `api: "openai-completions"`：

```json
{
  "providers": {
    "deepseek-web": {
      "baseUrl": "http://127.0.0.1:8787/v1",
      "api": "openai-completions",
      "apiKey": "$DEEPSEEK_WEB_API_KEY",
      "compat": {
        "supportsDeveloperRole": true,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": false,
        "supportsStore": false
      },
      "models": [
        {"id": "deepseek-v4-flash", "reasoning": true, "input": ["text"]},
        {"id": "deepseek-v4-pro", "reasoning": true, "input": ["text"]}
      ]
    }
  }
}
```

Responses 路径、thinking level、AGENTS.md/skills、MCP-backed tools、tool-call 历史和多轮行为见 **[docs/pi.md](docs/pi.md)**。该文档提供两套可复制的完整 `models.json` 配置。

## 多轮 session

继续同一会话有三种方式：

1. 传回本服务返回的 `previous_response_id`；
2. 传回 `conversation` / `chat_session_id`；
3. 像 Pi 一样发送完整稳定历史，由 `SessionStore` 匹配。

服务端保存 DeepSeek session 和最后一个可信 `response_message_id`。会话历史按**完整展开形态**落盘（`user → assistant(tool_calls) → tool(result) → assistant`），但匹配指纹会对展开形态做**折叠**：忽略 `tool`/`function` 轮并合并相邻 assistant，因此新版展开历史与旧版单条 canonical 历史都能命中同一 session，旧的 canonical 记录会在下一次对话时自动升级为展开形态。

请求超过 40 轮时按最近的 **user 边界**截断，保证回放历史始终以用户消息开头。切换公开模型或 thinking 开关时，只要历史或显式 ID 匹配，就继续同一 DeepSeek session。

> 同一 session 不支持并发写入。并行请求可能复用同一个 parent 并在上游形成分支。

## Tool calling 说明

DeepSeek Web 没有 OpenAI 原生 function calling。本项目把 tools/functions 和选择策略写入 prompt，并要求模型输出：

```text
<tool_call>
{"name":"tool_name","arguments":{"key":"value"}}
</tool_call>
```

解析成功后映射为标准结构。工具结果轮也会进入 session/history 匹配。

真实模型输出经常偏离上面的严格格式，解析器会做窄范围恢复，且不会把普通正文误判为调用：

- 接受松散的 ASCII 标签（`tool_call`、`tool-call`、`_call`、`call` 等）和原生 DSML 包装（`<｜｜DSML｜｜ invoke/parameter>`，全角竖线一到两根）。
- 纯 XML 形态（call + invoke + parameter 三层标签、无全角竖线、载荷不是 JSON 时）同样逐个恢复，标签在流式中隐藏；无 name 的 invoke 和没有 invoke 归属的孤立 parameter 不会被猜成调用。
- DSML 竖线与关键字之间是不间断空格（NBSP）、全角空格等 Unicode 空白时同样识别；相邻两个 DSML 包装之间的纯空白间隙在流式中静默隐藏，不会把协议原文或空白噪声帧下发给客户端。
- 包装标签不配对时（`<tool_call>` 开了没关、DSML invoke 没有 wrapper 开头却以 `</...calls>` 收尾）仍逐个恢复调用，孤立的关闭标签残差按协议垃圾隐藏；`</3` 这类散文片段不会被误吞。
- 开标签名漂移到 THINK、只留下 `<> ` 开头时，只要后面的工具 JSON 和闭合标签完整，仍按畸形开标签恢复，`<> ` 不会下发。
- 模型省略 `name`、只给出 `{"arguments":{...}}` 时，用注册工具的 `paramKeys` 唯一匹配推断工具名；两个工具都能解释全部参数时不猜。
- 严格解析失败且字符串含盘符前缀（如 `f:\workspace\...`）时，按 Windows 单反斜杠转义错误重试一次，修复后被丢弃的调用可以正常恢复。即使严格解析"成功"也会消歧：路径字符串里的 `\t`、`\b`、`\f`、`\r`、`\n` 会被 JSON 当成合法转义（如 `d:\tmp` 被静默改成制表符），此时只在**含盘符前缀的字符串内**把这些转义还原为字面反斜杠；不含盘符的字符串（如 `"a\tb"`）不受影响。
- 模型用 `"tool"` 字段（而非 `"name"`）写工具名时同样识别；DSML wrapper 开标签后直接放裸 JSON、invoke 开标签整个丢失（闭合标签依次错位为 parameter、invoke、calls）时，裸 JSON 仍恢复为调用，wrapper 整体隐藏；wrapper 内含正常 invoke 时不做此恢复。
- 仍失败时，对字符串里未转义的双引号（`python -c "..."` 命令、Write 的 `content` 内嵌 Python 三引号源码等）做最后手段的结构化解析：枚举引号的所有候选收尾并回溯，只接受能从 `{` 完整消费整段的解析；多个解析都成立时保留结构最多、不把 JSON 坍缩进字符串的那个；`{not json}` 这类非 JSON 不会被猜成调用。
- 流式过程中跨 chunk 的标签、孤立闭合标签和残缺 DSML 片段会被隐藏，不会以原始协议文本下发给客户端。
- 一个回合连续输出多个 bare 对象（空格或逗号分隔、同一行也可以，无论在 RESPONSE 通道还是漏进 THINK）都会逐个恢复并分配连续 index；对象之间的空白/逗号作为协议残差隐藏，散文夹在中间时不会跨过正文误收。
- 模型把 agent 客户端自己的工具结果包装（`<tool_call_result>`、`<toolcall_status>`、`<command_id>`、`<command_status>`、`<command_run_logs>` 等）整段当成回答吐出时，包装块（含未闭合的残缺块、丢失外层包装的内部标签）在流式中整段隐藏，并按空响应规则切新会话/压缩重试一次；块前的正常散文照常下发，普通 `<result>` 元素不会被误判。
- 上游在 SSE 内部返回错误帧（HTTP 仍是 200：`event: error`、`o:"ERROR"`、非零 `code` 或嵌套 `error` 记录，常见于内容风控、会话失效）时，返回 502 并带上 DeepSeek 的原始报错信息，不会被当成"空响应"消耗掉那一次自动重试。其中**上下文超限错误帧**（`finish_reason:"context_length_exceeded"` 或"达到对话长度上限/请开启新对话"文案）不会直接报错，而是按下方空响应的规则自动切新会话（非复用会话则原地压缩）重试一次；重试仍超限时才返回 502。
- 上游静默返回空 SSE（没有错误帧）且当前会话是复用会话时（常见于 agent 长会话累积历史超出网页端上下文：网页版上下文小于官方 API 的 128K），自动重试会切到**全新 DeepSeek 会话**并用压缩版 prompt 重发，而不是继续打向已膨胀的会话；非复用会话的空响应仍在当前会话内重试。Chat、Responses 两条链路以及无工具的普通请求都适用。相关日志附带 `upstreamTrace`（上游最后 6 个原始帧）便于事后取证，prompt 超过 13 万字符时会提前打 WARN 预警。
- prepare 阶段（PoW 挑战或会话创建）被上游以 `code:40003`、"Authorization Failed (invalid token)" 拒绝时（常见于服务长期运行期间登录态过期），自动清空缓存认证并重新校验登录（`data/auth.json` 仍有效则静默恢复，否则打开浏览器等待重新登录），随后用新登录态重新 prepare 并重放同一请求，仅重试一次；刷新后仍 40003 时才把鉴权错误返回给客户端。

这仍是**提示词模拟**：不能保证模型一定调用工具或严格遵守 JSON Schema。带工具的流式请求可能在尾部集中输出结构化 call，因为服务必须先清理协议文本。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | HTTP 监听地址；不建议改为公网地址 |
| `PORT` | `8787` | HTTP 端口 |
| `DS_CDP` | `http://127.0.0.1:9333` | Chrome DevTools Protocol endpoint |
| `DS_DATA_DIR` | `./data` | 运行时凭据、session、profile 根目录 |
| `DS_API_KEY` | 空 | 一个或多个 API key，逗号/空白分隔；优先于文件 |
| `DS_API_KEY_FILE` | `data/.api-key` | API key 文件，每行一个 |
| `DS_AUTH_FILE` | `data/auth.json` | DeepSeek token/cookie 快照 |
| `DS_SESSION_FILE` | `data/sessions.json` | session lineage 索引 |
| `DS_CHROME_PROFILE` | `data/chrome-profile` | 专用 Chrome profile |
| `DS_CHROME_PATH` | 自动发现 | Chrome/Chromium 可执行文件 |
| `DS_SHOW_BROWSER` | `false` | 日常请求显示浏览器；登录需要时始终可见 |
| `DS_POW_JS` | 内置当前 worker URL | DeepSeekHashV1 worker chunk URL |
| `DS_BASE_URL` | `https://chat.deepseek.com` | 上游地址，主要用于调试 |
| `DS_DEBUG` | `false` | 增加调试日志和 HTTP error stack |
| `DS_TOOL_REASONING` | `raw` | 工具轮思考可见性：`raw`（实时下发原始思考）、`clean`（实时下发并剔除泄漏的工具标签）、`hidden`（不下发，纯思考轮提升为正文） |

可复制 `.env.example` 为 `.env`。启动时读取该文件，但不会覆盖已存在的进程环境变量。

## 安全

以下路径包含凭据或私人会话内容，已由 `.gitignore` 排除：

```text
.env
data/auth.json
data/.api-key
data/sessions.json
data/chrome-profile/
```

- 不要提交、上传、打包、截图或粘贴这些内容。
- 保持 loopback 监听；API key 不是充分的公网安全边界。
- 如确需跨主机访问，使用受控网络、TLS reverse proxy、来源限制和独立密钥轮换。
- 认证数据泄漏可能等价于 DeepSeek 登录会话泄漏。

安全披露与账号风险见 [SECURITY.md](SECURITY.md)。

## 文档

- [架构与请求链路](docs/architecture.md)
- [API 兼容矩阵](docs/api.md)
- [Responses API 事件与对象](docs/responses-api.md)
- [Pi 完整集成指南](docs/pi.md)
- [故障排查](docs/troubleshooting.md)
- [维护者规格](SPEC.md)
- [贡献指南](CONTRIBUTING.md)
- [变更记录](CHANGELOG.md)

## 开发与验证

```bash
pnpm install --frozen-lockfile
pnpm typecheck
pnpm lint
pnpm test
pnpm build
```

CI 在 Node.js 20 和 22 上执行相同检查。单元测试不访问真实 DeepSeek 账号，也不在 CI 启动登录 e2e。

项目约束：

- TypeScript ESM，Node.js >= 20
- `node:http`，不引入重型 Web 框架
- 每个 `src/**/*.ts` 文件不超过 300 行
- strict TypeScript；不使用 `any` 绕过边界类型
- 协议行为改动必须有最小回归测试

## 已知限制

- 依赖私有 Web API、前端 PoW worker 和账号风控，可能无预告失效。
- 仅支持文本；不支持图片、文件、音频、视频或上传。
- 工具调用为 prompt 兼容层，不是原生 function calling。
- OpenAI `store`、后台 Responses、严格 structured output、hosted tools 和 prompt caching 未实现。
- 输入 token 无可靠上游计数，返回 `0`；输出 token 使用上游累计值。
- 单 session 并发请求可能分叉。

## Contributing

欢迎 focused issue/PR。提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)，并确保完整验证通过。报告日志前必须移除所有 token、cookie、key、账号和私人路径。

## License

[MIT](LICENSE) © 2026 lvanla
