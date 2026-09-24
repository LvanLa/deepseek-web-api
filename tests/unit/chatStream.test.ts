/** Verifies live Chat Completions mapping for plain and tool-compatible turns. */
import { describe, expect, it } from "vitest";

import {
  finalChunk,
  streamPlainChat,
  streamToolChat,
  toolSessionText,
  type ChatRun,
  type ChatStreamChunk,
} from "../../src/deepseek/chatStream.js";
import { EMPTY_TOOL_RESPONSE_TEXT } from "../../src/deepseek/toolOutcome.js";

type Channel = "reasoning" | "output";
type Delta = [Channel, string];

/** Build a synthetic upstream SSE body, one APPEND event per delta. */
function upstream(deltas: Delta[]): Response {
  const body =
    'event: ready\ndata: {"response_message_id":7,"request_message_id":6}\n\n' +
    deltas.map(([channel, text]) => {
      const fragment = { type: channel === "reasoning" ? "THINK" : "RESPONSE", content: text };
      return `data: ${JSON.stringify({ p: "response/fragments", o: "APPEND", v: [fragment] })}\n\n`;
    }).join("");
  return new Response(body);
}

function run(stream: Response): ChatRun {
  return { upstream: stream, publicModel: "deepseek-v4-pro", sessionId: "session-1" };
}

function collect(): { chunks: ChatStreamChunk[]; push: (chunk: ChatStreamChunk) => void } {
  const chunks: ChatStreamChunk[] = [];
  return { chunks, push: (chunk) => void chunks.push(chunk) };
}

function content(chunks: ChatStreamChunk[]): string {
  return chunks
    .map((chunk) => chunk.choices[0]?.delta.content)
    .filter((value): value is string => typeof value === "string")
    .join("");
}

function reasoning(chunks: ChatStreamChunk[]): string {
  return chunks
    .map((chunk) => chunk.choices[0]?.delta.reasoning_content)
    .filter((value): value is string => typeof value === "string")
    .join("");
}

function toolCalls(chunks: ChatStreamChunk[]) {
  return chunks.flatMap((chunk) => {
    const value = chunk.choices[0]?.delta.tool_calls;
    return Array.isArray(value) ? value : [];
  });
}

function kinds(chunks: ChatStreamChunk[]): string[] {
  return chunks.map((chunk) => {
    const delta = chunk.choices[0]?.delta ?? {};
    if ("reasoning_content" in delta) return "reasoning";
    if ("content" in delta) return "content";
    if ("tool_calls" in delta) return "tool";
    return "other";
  });
}

describe("streamToolChat", () => {
  it("promotes a hidden reasoning-only turn into one live content frame", async () => {
    const sink = collect();
    const value = await streamToolChat(
      run(upstream([["reasoning", "继续读取 src 目录。"]])),
      "hidden",
      "",
      sink.push,
    );

    expect(reasoning(sink.chunks)).toBe("");
    expect(content(sink.chunks)).toBe("继续读取 src 目录。");
    expect(kinds(sink.chunks)).toEqual(["content"]);
    expect(sink.chunks[0]?.choices[0]?.delta.role).toBe("assistant");
    expect(value.framesEmitted).toBe(1);
    expect(value.finishReason).toBe("stop");
    // The terminal finish chunk is emitted by the caller, not the stream helper.
    expect(sink.chunks.every((chunk) => chunk.choices[0]?.finish_reason === null)).toBe(true);
    sink.push(finalChunk(run(new Response()), 0, value.finishReason));
    expect(sink.chunks.at(-1)?.choices[0]?.finish_reason).toBe("stop");
  });

  it("emits the visible fallback for an empty turn", async () => {
    const sink = collect();
    const value = await streamToolChat(
      run(upstream([])),
      "hidden",
      EMPTY_TOOL_RESPONSE_TEXT,
      sink.push,
    );

    expect(content(sink.chunks)).toBe(EMPTY_TOOL_RESPONSE_TEXT);
    expect(kinds(sink.chunks)).toEqual(["content"]);
    expect(value.framesEmitted).toBe(1);
    expect(value.finishReason).toBe("stop");
  });

  it("streams a repaired tool call as tool_calls with no prose", async () => {
    const output = '<_call>\n{"name":"read","arguments":{"path":"src/index.ts"}\n</tool_call>';
    const sink = collect();
    const value = await streamToolChat(run(upstream([["output", output]])), "hidden", "", sink.push);

    expect(content(sink.chunks)).toBe("");
    expect(kinds(sink.chunks)).toEqual(["tool"]);
    expect(toolCalls(sink.chunks)).toHaveLength(1);
    expect(toolCalls(sink.chunks)[0]).toMatchObject({
      index: 0,
      function: { name: "read", arguments: '{"path":"src/index.ts"}' },
    });
    expect(value.finishReason).toBe("tool_calls");
  });

  it("never streams the mangled < calls> residue of a DSML run_code turn", async () => {
    const output =
      '< calls> <｜｜DSML｜｜invoke name="run_code">' +
      '<｜｜DSML｜｜parameter name="code" string="true">ls</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>';
    const sink = collect();
    const value = await streamToolChat(run(upstream([["output", output]])), "hidden", "", sink.push);

    expect(content(sink.chunks)).toBe("");
    expect(JSON.stringify(sink.chunks)).not.toContain("< calls>");
    expect(toolCalls(sink.chunks)[0]?.function.name).toBe("run_code");
    expect(value.finishReason).toBe("tool_calls");
    const sessionText = toolSessionText(value.outcome);
    expect(sessionText).not.toContain("< calls>");
    expect(sessionText).not.toContain("｜");
    expect(sessionText).toContain("<tool_call>");
  });

  it("forwards raw reasoning deltas before answer deltas", async () => {
    const sink = collect();
    await streamToolChat(
      run(upstream([
        ["reasoning", "思"],
        ["reasoning", "考中"],
        ["output", "最终答案"],
      ])),
      "raw",
      "",
      sink.push,
    );

    expect(kinds(sink.chunks)).toEqual(["reasoning", "reasoning", "content"]);
    expect(reasoning(sink.chunks)).toBe("思考中");
    expect(content(sink.chunks)).toBe("最终答案");
  });

  it("releases safe prose before a tool block that closes in a later chunk", async () => {
    const sink = collect();
    await streamToolChat(
      run(upstream([
        ["output", "先读取文件。"],
        ["output", '<tool_call>{"name":"read"'],
        ["output", ',"arguments":{"path":"a.ts"}}</tool_call>'],
      ])),
      "hidden",
      "",
      sink.push,
    );

    expect(kinds(sink.chunks)).toEqual(["content", "tool"]);
    expect(content(sink.chunks)).toBe("先读取文件。");
    expect(toolCalls(sink.chunks)[0]?.function).toMatchObject({
      name: "read",
      arguments: '{"path":"a.ts"}',
    });
  });

  it("strips leaked tool protocol from clean-mode reasoning and recovers the call", async () => {
    const sink = collect();
    const value = await streamToolChat(
      run(upstream([
        ["reasoning", "Need data.\n"],
        ["reasoning", '<tool_call>{"name":"bash"'],
        ["reasoning", ',"arguments":{"command":"date"}}</tool_call>'],
      ])),
      "clean",
      "",
      sink.push,
    );

    expect(reasoning(sink.chunks)).toBe("Need data.\n");
    expect(JSON.stringify(sink.chunks)).not.toContain("<tool_call>");
    expect(toolCalls(sink.chunks)).toHaveLength(1);
    expect(toolCalls(sink.chunks)[0]?.function.name).toBe("bash");
    expect(value.finishReason).toBe("tool_calls");
  });

  it("assigns continuous indexes to multiple calls from one turn", async () => {
    const first = '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>';
    const second = '<tool_call>{"arguments":{"command":"pwd"},"name":"bash"}</tool_call>';
    const sink = collect();
    await streamToolChat(
      run(upstream([["output", first], ["output", second]])),
      "hidden",
      "",
      sink.push,
    );

    const calls = toolCalls(sink.chunks);
    expect(calls.map((call) => call.function?.name)).toEqual(["read", "bash"]);
    expect(calls.map((call) => call.index)).toEqual([0, 1]);
  });

  it("recovers three space-separated bare objects streamed character by character", async () => {
    const text = [
      '{"name":"Grep","arguments":{"pattern":"GenerateActivityResponse"}}',
      '{"name":"Grep","arguments":{"pattern":"activity_id","path":"a"}}',
      '{"name":"Grep","arguments":{"pattern":"activity_id","path":"b"}}',
    ].join(" ");
    const deltas: Delta[] = [...text].map((char) => ["output", char]);
    const sink = collect();
    const value = await streamToolChat(run(upstream(deltas)), "hidden", "", sink.push);

    const calls = toolCalls(sink.chunks);
    expect(calls).toHaveLength(3);
    expect(calls.map((call) => call.index)).toEqual([0, 1, 2]);
    expect(content(sink.chunks)).toBe("");
    expect(value.finishReason).toBe("tool_calls");
  });

  it("recovers three comma-separated bare objects leaked into THINK character by character", async () => {
    const text = [
      '{"name":"Grep","arguments":{"pattern":"GenerateActivityResponse"}}',
      '{"name":"Grep","arguments":{"pattern":"activity_id","path":"a"}}',
      '{"name":"Grep","arguments":{"pattern":"activity_id","path":"b"}}',
    ].join(",");
    const deltas: Delta[] = [...text].map((char) => ["reasoning", char]);
    const sink = collect();
    const value = await streamToolChat(run(upstream(deltas)), "hidden", "", sink.push);

    const calls = toolCalls(sink.chunks);
    expect(calls).toHaveLength(3);
    expect(content(sink.chunks)).toBe("");
    expect(value.finishReason).toBe("tool_calls");
  });
});

describe("streamPlainChat", () => {
  it("streams reasoning and answer deltas live and closes with stop", async () => {
    const sink = collect();
    const result = await streamPlainChat(
      run(upstream([
        ["reasoning", "核对事实"],
        ["output", "答案正文"],
      ])),
      sink.push,
    );

    // The last chunk is the terminal finish frame with an empty delta.
    expect(kinds(sink.chunks).slice(0, -1)).toEqual(["reasoning", "content"]);
    expect(reasoning(sink.chunks)).toBe("核对事实");
    expect(content(sink.chunks)).toBe("答案正文");
    expect(result.finalText).toBe("答案正文");
    expect(sink.chunks.at(-1)?.choices[0]?.finish_reason).toBe("stop");
  });

  it("keeps a reasoning-only turn on the reasoning channel without duplicating it", async () => {
    const sink = collect();
    const result = await streamPlainChat(
      run(upstream([["reasoning", "只有思考"]])),
      sink.push,
    );

    expect(kinds(sink.chunks).slice(0, -1)).toEqual(["reasoning"]);
    expect(content(sink.chunks)).toBe("");
    expect(reasoning(sink.chunks)).toBe("只有思考");
    expect(result.finalText).toBe("只有思考");
    expect(sink.chunks.at(-1)?.choices[0]?.finish_reason).toBe("stop");
  });
});
