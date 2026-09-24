/** Verifies the documented behavior of the corresponding production module. */
import { describe, expect, it } from "vitest";

import { SseParser, iterDeepSeekSse } from "../../src/deepseek/sse.js";
import { iterDeepSeekUpdates } from "../../src/deepseek/updates.js";
import { HttpError } from "../../src/utils/errors.js";

describe("DeepSeek SSE parser", () => {
  it("parses chunk boundaries, event names, and CRLF", () => {
    const parser = new SseParser();
    expect(parser.push('event: ready\r\ndata: {"response_message_id":')).toEqual([]);
    expect(parser.push('42}\r\n\r\ndata: {"v":"hello"}\n\n')).toEqual([
      { event: "ready", data: { response_message_id: 42 }, raw: '{"response_message_id":42}' },
      { event: null, data: { v: "hello" }, raw: '{"v":"hello"}' },
    ]);
  });

  it("ignores malformed JSON and parses a final unterminated block", () => {
    const parser = new SseParser();
    parser.push("data: not-json\n\n");
    parser.push('event: title\ndata: {"content":"name"}');
    expect(parser.finish()).toEqual([
      { event: "title", data: { content: "name" }, raw: '{"content":"name"}' },
    ]);
  });

  it("iterates a web Response stream", async () => {
    const response = new Response('event: ready\ndata: {"request_message_id":1}\n\n');
    const events = [];
    for await (const event of iterDeepSeekSse(response)) events.push(event);
    expect(events).toEqual([
      { event: "ready", data: { request_message_id: 1 }, raw: '{"request_message_id":1}' },
    ]);
  });

  it("keeps reasoning and output deltas separate", async () => {
    const body = [
      'data: {"p":"response/fragments","o":"APPEND","v":[{"type":"THINK","content":"why"}]}\n\n',
      'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"?"}\n\n',
      'data: {"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"answer"}]}\n\n',
      'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"!"}\n\n',
    ].join("");
    const updates = [];
    for await (const update of iterDeepSeekUpdates(new Response(body))) updates.push(update);
    expect(updates).toEqual([
      { type: "reasoning", delta: "why" },
      { type: "reasoning", delta: "?" },
      { type: "output", delta: "answer" },
      { type: "output", delta: "!" },
    ]);
  });

  it("reports each raw frame through the trace sink", async () => {
    const body =
      'event: ready\ndata: {"response_message_id":7}\n\ndata: {"v":"drip"}\n\n';
    const seen: Array<{ event: string | null; raw: string }> = [];
    const updates = [];
    for await (const update of iterDeepSeekUpdates(new Response(body), (entry) => seen.push(entry))) {
      updates.push(update);
    }
    expect(updates).toHaveLength(2);
    expect(seen).toEqual([
      { event: "ready", raw: '{"response_message_id":7}' },
      { event: null, raw: '{"v":"drip"}' },
    ]);
    expect(updates[1]).toEqual({ type: "reasoning", delta: "drip" });
  });
});

describe("DeepSeek upstream error frames", () => {
  async function collect(body: string) {
    const updates = [];
    for await (const update of iterDeepSeekUpdates(new Response(body))) updates.push(update);
    return updates;
  }

  it("surfaces an event:error frame with its upstream message", async () => {
    const body = 'event: error\ndata: {"message":"rate limited by upstream"}\n\n';
    await expect(collect(body)).rejects.toMatchObject({ status: 502 });
    await expect(collect(body)).rejects.toThrow("rate limited by upstream");
    await expect(collect(body)).rejects.toBeInstanceOf(HttpError);
  });

  it("surfaces ERROR-op patches and nested error records", async () => {
    const body =
      'data: {"p":"response","o":"ERROR","v":{"errmsg":"content policy rejected prompt"}}\n\n';
    await expect(collect(body)).rejects.toThrow("content policy rejected prompt");
    const nested = 'data: {"error":{"message":"context length exceeded"}}\n\n';
    await expect(collect(nested)).rejects.toThrow("context length exceeded");
  });

  it("treats nonzero code plus message as an error but code 0 stays benign", async () => {
    const bad = 'data: {"code":50021,"msg":"session unavailable"}\n\n';
    await expect(collect(bad)).rejects.toThrow("session unavailable");
    // A success-style frame with code 0 must not be mistaken for an error.
    const ok = 'data: {"code":0,"v":{"hello":1}}\n\n';
    await expect(collect(ok)).resolves.toEqual([]);
  });

  it("marks a context_length_exceeded frame so the client can fork", async () => {
    const body = 'data: {"type":"error","content":"达到对话长度上限，请开启新对话",' +
      '"clear_response":true,"finish_reason":"context_length_exceeded"}\n\n';
    await expect(collect(body)).rejects.toMatchObject({
      status: 502, code: "context_length_exceeded",
    });
    await expect(collect(body)).rejects.toThrow("达到对话长度上限");
    // Text signal without an explicit finish_reason is classified the same.
    const textOnly = 'data: {"type":"error","message":"对话长度达到上限，请开启新对话"}\n\n';
    await expect(collect(textOnly)).rejects.toMatchObject({ code: "context_length_exceeded" });
  });
});
