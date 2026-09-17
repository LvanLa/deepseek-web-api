---
name: tool-protocol-recovery
description: Recovers malformed tool-call protocol (tool tags, DSML, bare JSON) leaking verbatim into streams in this repo. Use when calls are missing, tags are shown to users, or frames arrive only at turn end. Not for HTTP or auth failures.
---

# Tool Protocol Recovery

Diagnose and repair cases where the model's tool-call text is not converted into
structured calls: tags/JSON/DSML appear verbatim in the answer, the call count is
lower than the model intended, or a tools request only emits frames at turn end.

## 1. Reproduce with evidence before editing

Create a temporary test (for example `tests/unit/_repro.test.ts`), feed the exact
user sample through BOTH surfaces, then delete it afterwards:

- whole text and closed vs unclosed variants through `parseToolCalls(text)`;
- whole feed plus tiny chunks (slice every 10-20 characters) through `StreamSieve`
  (`feed` per chunk, then `flush`).

Log returned `content`, call count/names/arguments, and whether chunked output
produces text events. Never assume the parser or the streaming layer is at fault:
the whole-text path and the chunked path fail for different reasons.

## 2. Locate the failing layer (upstream Sieve -> mapping -> client)

Walk the text through these stages in order; the defect is typically one of them:

1. Candidate collection - `collectJsonObjects` in `toolCallJson.ts`. Default mode
   only accepts objects whose first key is `name`, `arguments`, or `function`.
   Inside a tool-call tag the first standalone object must always be a candidate.
2. Open/close pairing - `dsmlElements` in `dsmlToolCalls.ts`. Malformed streams
   close a parameter with an invoke close tag, omit the invoke opener, or emit
   duplicate closers. Parameters must accept both close kinds; orphan parameters
   are rebuilt per parameter, reusing the nearest preceding invoke name.
3. Payload shaping - `callPayload` in `toolCalls.ts`. Handles the
   `{name, arguments}` envelope, name-first bare JSON, and the unwrapped shell
   shape (`{"command":"..."}` with no name) which maps to `bash`, keeping only
   `command` and `cwd` so strict schemas do not see harness metadata.
4. Streaming capture - `StreamSieve` in `streamSieve.ts`.
   - `couldStartTag` must hold partial openers across chunks, including the
     whitespace between the second bar group and the DSML keyword; ordinary
     prose such as `a < b` must still be released.
   - `captureComplete` inside a wrapper completes only on the wrapper close,
     never when an individual invoke closes; close detection requires the full
     `>` character.
   - `flush` must also parse an unclosed tag capture, so truncated but
     recognizable calls are recovered; irreparable garbage stays hidden and is
     left to the empty-turn fallback/retry.
5. Framing/gating - `streamToolChat`/`streamPlainChat` in `chatStream.ts` and the
   zero-frame retry gate in `client.ts` (retry only when no frame was emitted).

## 3. Fix minimally and lock it in

- Change only the failing layer; do not alter unrelated behavior or heuristics.
- Keep recovery heuristics narrow and add a negative case proving ordinary prose
  or harmless JSON is not mistaken for a call.
- Add a permanent regression test next to existing ones:
  `tests/unit/toolCalls.test.ts` for parse shapes,
  `tests/unit/streamSieve.test.ts` for streaming/flush behavior,
  `tests/unit/mapResponses.test.ts` for Responses live event ordering.

## 4. Verify

Run from the repo root:

- `node scripts/check-source-lines.mjs` - every file under `src` must be at most
  300 physical lines; compress before finishing.
- `node scripts/check-comments.mjs` - code comments must be ASCII English.
- `node node_modules/typescript/bin/tsc -p tsconfig.json --noEmit` and the same
  with `tsconfig.build.json` (exactOptionalPropertyTypes is enabled; never pass
  explicit `undefined` for optional handler fields - use conditional spreads).
- `node node_modules/eslint/bin/eslint.js "src/**/*.ts" "tests/**/*.ts" vitest.config.ts`
- `node node_modules/vitest/vitest.mjs run`
- Finish with IDE diagnostics; they must be empty.

On this Windows sandbox pnpm can fail with disk I/O errors; invoke tsc, eslint,
and vitest through `node node_modules/...` directly, and if needed point
`TMP`/`TEMP` at a `.tmp` folder inside the workspace, then delete that folder.

## Key domain facts

- Parser entry point is `parseToolCalls(text, seed)`; the shared streaming core
  for Chat and Responses is `streamToolTurn` in `streamTurn.ts`.
- Tag families: loose ASCII `TOOL_TAG` (`tool_call`, `tool-call`, `toolcall`,
  `_call`, `call`), native DSML with one or two U+FF5C full-width bars
  (`<[bars]DSML[bars]calls/invoke/parameter>`), and protocol-only bare JSON at
  stream start. `looksLikeToolJson` requires both `name` and `arguments`.
- Chat frames use `reasoning_content`, `content`, `tool_calls`; the final chunk
  carries `finish_reason` of `stop` or `tool_calls`.
- `framesEmitted` counts handler callbacks; `client.ts` retries silently only
  when the turn was recoverable and zero frames were emitted.
