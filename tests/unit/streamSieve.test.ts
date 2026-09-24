/** Verifies incremental separation of visible prose from tool-call blocks. */
import { describe, expect, it } from "vitest";

import { StreamSieve, type SieveEvent } from "../../src/deepseek/streamSieve.js";
import { parseToolCalls } from "../../src/deepseek/toolCalls.js";

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

  it("hides whitespace between two adjacent wrappers fed one char at a time", () => {
    const wrapper = (name: string) => [
      "<｜｜DSML｜｜ calls>",
      `<｜｜DSML｜｜ invoke name="${name}">`,
      `<｜｜DSML｜｜ parameter name="path" string="true">${name.toLowerCase()}</｜｜DSML｜｜ parameter>`,
      "</｜｜DSML｜｜ invoke>",
      "</｜｜DSML｜｜ calls>",
    ].join("");
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(`${wrapper("LS")}    ${wrapper("Read")}`, 1));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual(["LS", "Read"]);
  });

  it("captures DSML whose label gaps are NBSP instead of regular spaces", () => {
    const gap = " ";
    const text = [
      `<｜｜DSML｜｜${gap}calls>`,
      `<｜｜DSML｜｜${gap}invoke name="LS">`,
      `<｜｜DSML｜｜${gap}parameter name="path" string="true">src</｜｜DSML｜｜parameter>`,
      `</｜｜DSML｜｜${gap}invoke>`,
      `</｜｜DSML｜｜${gap}calls>`,
    ].join("");
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(text, 2));
    expect(textDeltas(events)).toBe("");
    expect(sieve.calls).toHaveLength(1);
    expect(sieve.flush()).toEqual([]);
  });
});

describe("StreamSieve mismatched wrappers", () => {
  const mixed = String.raw`<tool_call>
{"name":"RunCommand","arguments":{"command":"node -v","blocking":true,"requires_approval":false,"command_type":"short_running_process"}}
<｜｜DSML｜｜ invoke name="RunCommand">
<｜｜DSML｜｜ parameter name="blocking" string="false">true</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="command" string="true">opdev help</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="command_type" string="true">short_running_process</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="requires_approval" string="false">false</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="wait_ms_before_async" string="false">0</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>`;

  it.each([1, 2, 5])("recovers both calls and hides orphan close tags at width %s", (w) => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(mixed, w));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual(["RunCommand", "RunCommand"]);
    expect(JSON.parse(toolCallsOf(events)[1]!.function.arguments)).toEqual({
      blocking: true,
      command: "opdev help",
      command_type: "short_running_process",
      requires_approval: false,
      wait_ms_before_async: 0,
    });
  });

  it("does not mistake a heart-like </3 fragment for a close tag", () => {
    const sieve = new StreamSieve();
    expect(sieve.feed("love </3 yeah")).toEqual([{ type: "text", delta: "love </3 yeah" }]);
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

  it("recovers a call when the opener name drifts into THINK, leaving <>", () => {
    const body = '{"name":"TodoWrite","arguments":{"merge":true,"todos":[]}}';
    const sieve = new StreamSieve("seed", [
      { name: "TodoWrite", paramKeys: ["todos", "merge"] },
    ]);
    const events = feedChunks(sieve, ["<", ...splitEvery(`>\n${body}\n</call>`, 12)]);
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(sieve.calls[0]?.function.name).toBe("TodoWrite");
  });

  it("recovers a command containing unescaped inner double quotes", () => {
    const body = '{"name":"RunCommand","arguments":{"command":"python -c "print(\'ok\')"","blocking":true}}';
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(`<tool_call>\n${body}\n</tool_call>`, 13));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(JSON.parse(sieve.calls[0]?.function.arguments ?? "{}")).toEqual({
      command: "python -c \"print('ok')\"",
      blocking: true,
    });
  });

  it("recovers a Write call whose content embeds triple-quoted source", () => {
    const embedded =
      '"""主题包 + Planner 单元测试。\n' +
      'THEME_FIELDS = {\n"theme_key", "festival", "title",\n}\n' +
      'payload = json.loads(prompt.split("\\n", 1)[1])\n' +
      'assert "base_missions" not in payload, "prompt 不能携带成稿任务"\n"';
    const body = '{"name":"Write","arguments":{"file_path":"f:\\workspace\\play-together\\backend\\tests\\t.py",' +
      `"content":"${embedded}"}}`;
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(`<tool_call>\n${body}\n</tool_call>`, 11));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(JSON.parse(sieve.calls[0]?.function.arguments ?? "{}")).toEqual({
      file_path: "f:\\workspace\\play-together\\backend\\tests\\t.py",
      content: embedded,
    });
  });

  const parallel = [
    '{"name":"read","arguments":{"path":"a.ts"}}',
    '{"name":"bash","arguments":{"command":"ls"}}',
    '{"name":"Grep","arguments":{"pattern":"x"}}',
  ];

  it("recovers space-separated bare objects one char at a time with no leaks", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(parallel.join(" "), 1));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual(["read", "bash", "Grep"]);
  });

  it("recovers comma-separated bare objects without leaking the commas", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(parallel.join(","), 3));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(3);
  });
});

describe("StreamSieve plain-XML invoke protocol", () => {
  const openTag = (tag: string, attrs = ""): string => "<" + tag + attrs + ">";
  const closeTag = (tag: string): string => "</" + tag + ">";
  const parameter = (name: string, body: string): string =>
    openTag("parameter", ` name="${name}"`) + body + closeTag("parameter");

  it("hides a call-wrapped invoke fed in tiny chunks and emits its tool call", () => {
    const path = "c:\\Users\\研发部\\plugins\\lark\\1.0.5\\lark-base-workflow-schema.md";
    const stanza = openTag("call") + " " + openTag("invoke", ' name="Read"') + " " +
      parameter("file_path", path) + closeTag("invoke") + " " + closeTag("call");
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(stanza, 7));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(sieve.calls[0]?.function.name).toBe("Read");
    expect(sieve.calls[0]?.function.arguments).toBe(JSON.stringify({ file_path: path }));
  });

  it("captures a standalone invoke without a call wrapper", () => {
    const stanza = openTag("invoke", ' name="Bash"') + parameter("command", "ls -la") + closeTag("invoke");
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(stanza, 11));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toHaveLength(1);
    expect(sieve.calls[0]?.function.name).toBe("Bash");
  });
});

describe("StreamSieve tool-result blocks", () => {
  const block = "<tool_call_result><toolcall_status>Done</toolcall_status>" +
    "<command_id>job-7c9a</command_id>" +
    "<command_run_logs>81 passed in 2.51s</command_run_logs></tool_call_result>";

  it("hides a result block fed in tiny chunks and stays empty on flush", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(block, 3));
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events)).toEqual([]);
    expect(sieve.flush()).toEqual([]);
    expect(parseToolCalls(sieve.fullOutput.join(""))).toEqual({ content: "", toolCalls: [] });
  });

  it("releases prose preceding a result block but hides the block", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, ["Done.", block]);
    expect(textDeltas(events)).toBe("Done.");
    expect(sieve.flush()).toEqual([]);
  });
});

describe("StreamSieve tool key and malformed DSML", () => {
  const winPath = String.raw`d:\tmp\tijian`;
  const turn = `<_call>\n{"tool": "LS", "arguments": {"path": "${winPath}"}}\n</_call>\n` +
    `<｜｜DSML｜｜ calls>\n{"tool": "Grep", "arguments": {"pattern": "M6", ` +
    `"path": "${winPath}", "output_mode": "files_with_matches"}}\n` +
    `</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>`;

  it("recovers both malformed calls char by char without leaking protocol", () => {
    const sieve = new StreamSieve();
    const events = feedChunks(sieve, splitEvery(turn, 1));
    events.push(...sieve.flush());
    expect(textDeltas(events)).toBe("");
    expect(toolCallsOf(events).map((call) => call.function.name)).toEqual(["LS", "Grep"]);
  });
});
