/** Verifies incremental separation of visible prose from tool-call blocks. */
import { describe, expect, it } from "vitest";

import { StreamSieve, type SieveEvent } from "../../src/deepseek/streamSieve.js";

function feedChunks(sieve: StreamSieve, chunks: string[]): SieveEvent[] {
  return chunks.flatMap((chunk) => sieve.feed(chunk));
}

function splitEvery(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let index = 0; index < text.length; index += size) {
    pieces.push(text.slice(index, index + size));
  }
  return pieces;
}

function textDeltas(events: SieveEvent[]): string {
  return events
    .filter((event) => event.type === "text")
    .map((event) => (event.type === "text" ? event.delta : ""))
    .join("");
}

function toolCallsOf(events: SieveEvent[]) {
  return events.flatMap((event) => (event.type === "toolCalls" ? event.calls : []));
}

describe("StreamSieve prose", () => {
  it("releases plain text immediately without holding a suffix", () => {
    const sieve = new StreamSieve();
    expect(sieve.feed("hello world")).toEqual([{ type: "text", delta: "hello world" }]);
    expect(sieve.feed(" more")).toEqual([{ type: "text", delta: " more" }]);
    expect(sieve.flush()).toEqual([]);
    expect(sieve.calls).toEqual([]);
  });

  it("does not mistake ordinary less-than prose for an opener", () => {
    const sieve = new StreamSieve();
    expect(sieve.feed("a < b")).toEqual([{ type: "text", delta: "a < b" }]);
    expect(sieve.feed("yum < cookies")).toEqual([{ type: "text", delta: "yum < cookies" }]);
    // A lone "<" is held, then released once the next chars disprove an opener.
    expect(sieve.feed("<")).toEqual([]);
    expect(sieve.feed("= 3")).toEqual([{ type: "text", delta: "<= 3" }]);
    expect(sieve.flush()).toEqual([]);
  });

  it("releases a disproved tag prefix at a chunk boundary", () => {
    const sieve = new StreamSieve();
    expect(sieve.feed("<ca")).toEqual([]);
    expect(sieve.feed("kes are sweet")).toEqual([
      { type: "text", delta: "<cakes are sweet" },
    ]);
    expect(sieve.flush()).toEqual([]);
  });
});

describe("StreamSieve tagged tool calls", () => {
  const tagged = '<tool_call>\n{"name":"read","arguments":{"path":"a"}}\n</tool_call>';

  it("captures a complete tag without leaking any prose", () => {
    const sieve = new StreamSieve();
    const events = sieve.feed(tagged);
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(sieve.calls[0]).toMatchObject({
      function: { name: "read", arguments: '{"path":"a"}' },
    });
    expect(sieve.flush()).toEqual([]);
  });

  it("holds a tag split across small chunks until it closes", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(tagged, 5));
    expect(textDeltas(events)).toBe("");
    expect(events).toHaveLength(1);
    expect(events[0]?.type).toBe("toolCalls");
    expect(sieve.calls).toHaveLength(1);
    expect(sieve.flush()).toEqual([]);
  });

  it("streams prose that precedes a split tool block first", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, ["先读取文件。", "<tool_call>", '{"name":"read"', "}</tool_call>"]);
    expect(events[0]).toEqual({ type: "text", delta: "先读取文件。" });
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(textDeltas(events)).toBe("先读取文件。");
  });

  it("captures multiple calls back to back in emission order", () => {
    const text = [
      '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
      '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
    ].join("\n");
    const sieve = new StreamSieve();
    const events = sieve.feed(text);
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual(["read", "bash"]);
    expect(textDeltas(events)).toBe("");
  });

  it("keeps global indexes when closed tags arrive in separate chunks", () => {
    const first = '<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>';
    const second = '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>';
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, [first, second]);
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual(["read", "bash"]);
  });

  it("flushes an unclosed block as plain text", () => {
    const sieve = new StreamSieve();
    expect(sieve.feed("<tool_call>not closed")).toEqual([]);
    expect(sieve.flush()).toEqual([{ type: "text", delta: "<tool_call>not closed" }]);
  });

  it("recovers an unclosed shell-command object at flush without leaking text", () => {
    const body = '{"command":"npx tsc --noEmit","blocking":true,"requires_approval":false}';
    const sieve = new StreamSieve();
    expect(feedChunks(sieve, splitEvery(`<tool_call>\n${body}`, 7))).toEqual([]);
    const events = sieve.flush();
    expect(events).toHaveLength(1);
    expect(events[0]?.type).toBe("toolCalls");
    expect(toolCallsOf(events)[0]).toMatchObject({
      function: { name: "bash", arguments: '{"command":"npx tsc --noEmit"}' },
    });
    expect(textDeltas(events)).toBe("");
  });

  it("force-releases a capture buffer that grows past the limit", () => {
    const sieve = new StreamSieve();
    const oversized = "<tool_call>" + "x".repeat(1_100_000);
    const events = sieve.feed(oversized);
    expect(events).toHaveLength(1);
    expect(events[0]?.type).toBe("text");
    expect(textDeltas(events)).toHaveLength(oversized.length);
    expect(sieve.calls).toEqual([]);
    expect(sieve.flush()).toEqual([]);
  });
});

describe("StreamSieve DSML", () => {
  const invoke =
    '<｜｜DSML｜｜invoke name="run_code">' +
    '<｜｜DSML｜｜parameter name="code" string="true">ls</｜｜DSML｜｜parameter>' +
    "</｜｜DSML｜｜invoke>";

  it("captures a full-width DSML invoke split across tiny chunks", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(invoke, 3));
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(sieve.calls[0]).toMatchObject({
      function: { name: "run_code", arguments: '{"code":"ls"}' },
    });
    expect(sieve.flush()).toEqual([]);
  });

  it("hides the mangled ASCII < calls> residue around a DSML invoke", () => {
    const residue = '< calls> ' + invoke;
    const sieve = new StreamSieve();
    const events = sieve.feed(residue);
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual(["run_code"]);
    expect(textDeltas(events)).toBe("");
  });

  it("holds a wrapped multi-invoke block split across chunks until wrapper close", () => {
    const wrapped = [
      "<｜｜DSML｜｜ calls>",
      '<｜｜DSML｜｜ invoke name="Glob">',
      '<｜｜DSML｜｜ parameter name="pattern" string="true"> /package.json</｜｜DSML｜｜ parameter>',
      '<｜｜DSML｜｜ parameter name="path" string="true">f:\\workspace\\play-together</｜｜DSML｜｜ parameter>',
      "</｜｜DSML｜｜ invoke>",
      '<｜｜DSML｜｜ invoke name="Glob">',
      '<｜｜DSML｜｜ parameter name="pattern" string="true"> /typings/**/*.d.ts</｜｜DSML｜｜ parameter>',
      '<｜｜DSML｜｜ parameter name="path" string="true">f:\\workspace\\play-together\\miniprogram</｜｜DSML｜｜ parameter>',
      "</｜｜DSML｜｜ invoke>",
      '<｜｜DSML｜｜ invoke name="Read">',
      '<｜｜DSML｜｜ parameter name="file_path" string="true">f:\\workspace\\play-together\\miniprogram\\pages\\memory\\memory.wxss</｜｜DSML｜｜ parameter>',
      "</｜｜DSML｜｜ invoke>",
      '<｜｜DSML｜｜ invoke name="LS">',
      '<｜｜DSML｜｜ parameter name="path" string="true">f:\\workspace\\play-together\\backend</｜｜DSML｜｜ parameter>',
      "</｜｜DSML｜｜ invoke>",
      "</｜｜DSML｜｜ calls>",
    ].join("\n");
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(wrapped, 10));
    expect(textDeltas(events)).toBe("");
    expect(events).toHaveLength(1);
    expect(events[0]?.type).toBe("toolCalls");
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual([
      "Glob", "Glob", "Read", "LS",
    ]);
    expect(sieve.calls).toHaveLength(4);
    expect(sieve.flush()).toEqual([]);
  });
});

describe("StreamSieve bare JSON", () => {
  it("commits a protocol-only bare object at the start of output", () => {
    const sieve = new StreamSieve();
    const events = sieve.feed('{"name":"read","arguments":{"path":"a"}}');
    expect(events[0]?.type).toBe("toolCalls");
    expect(sieve.calls[0]?.function.name).toBe("read");
    expect(textDeltas(events)).toBe("");
  });

  it("holds a split bare object closed across chunks", () => {
    const sieve = new StreamSieve();
    expect(sieve.feed('{"name":"read"')).toEqual([]);
    const events = sieve.feed(',"arguments":{"path":"a"}}');
    expect(events[0]?.type).toBe("toolCalls");
    expect(sieve.calls).toHaveLength(1);
  });

  it("releases a non-tool leading object and keeps streaming prose", () => {
    const sieve = new StreamSieve();
    expect(sieve.feed('{"city":"Hefei"}')).toEqual([
      { type: "text", delta: '{"city":"Hefei"}' },
    ]);
    expect(sieve.feed(" is nice")).toEqual([{ type: "text", delta: " is nice" }]);
    expect(sieve.calls).toEqual([]);
    expect(sieve.flush()).toEqual([]);
  });

  it("infers the tool name from hints and hides a trailing orphan DSML close", () => {
    const body = String.raw`{"arguments":{"pattern":"Options<","path":"f:\workspace\x","glob":"*.d.ts","output_mode":"content","-n":true,"head_limit":40}}`;
    const sample = `<_call>\n${body}\n</tool_call>\n</｜｜DSML｜｜ calls>`;
    const tools = [
      { name: "Glob", paramKeys: ["pattern", "path", "glob"] },
      { name: "Grep", paramKeys: ["pattern", "path", "glob", "output_mode", "-n", "head_limit"] },
    ];
    const sieve = new StreamSieve("seed", tools);
    const events = feedChunks(sieve, splitEvery(sample, 12));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(sieve.calls[0]?.function.name).toBe("Grep");
  });
});
