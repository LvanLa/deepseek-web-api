/** Verifies parsing of the text protocol into OpenAI-compatible tool calls. */
import { describe, expect, it } from "vitest";

import {
  canonicalAssistantText,
  canonicalParsedAssistantText,
  parseToolCalls,
  parseToolCallsFromParts,
} from "../../src/deepseek/toolCalls.js";

const FAILED_SESSION_A = `{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/.github/workflows/ci.yml"}

{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/tsconfig.json"}`;

const FAILED_SESSION_B = `<_call>
{"name": "bash", "arguments": {"command": "find /Users/kittors/Developer/opensource/deepseek-web-api/src -name '*.ts' | sort", "timeout": 5}
</tool_call>`;

const FAILED_SESSION_C = `<_call>
{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/client.ts"}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/src/server/routes.ts"}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/completion.ts"}
</tool_call>`;

describe("parseToolCalls", () => {
  it("maps tagged JSON to OpenAI tool_calls while preserving normal content", () => {
    const result = parseToolCalls(
      'Checking first.\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Hefei"}}\n</tool_call>',
      "response-1",
    );

    expect(result.content).toBe("Checking first.");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]).toMatchObject({
      id: expect.stringMatching(/^call_[0-9a-f]{24}$/),
      type: "function",
      function: { name: "get_weather", arguments: '{"city":"Hefei"}' },
    });
  });

  it("leaves malformed call text as ordinary content", () => {
    const text = "<tool_call>{not json}</tool_call>";
    expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
  });

  it("leaves pure prose unchanged", () => {
    const text = 'Use JSON like {"name":"example"} when documenting payloads.';
    expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
  });

  it("accepts attribute-name tags used by DeepSeek/Pi hybrid output", () => {
    const sample = `<_call>
<tool_call name="read">
{"arguments": {"path": "/tmp/README.md", "limit": 100}}
</tool_call>
<tool_call name="read">
{"arguments": {"path": "/tmp/SPEC.md", "limit": 100}}
</tool_call>
<tool_call name="read">
{"arguments": {"arguments": {"path": "/tmp/package.json"}}}
</tool_call>
<tool_call name="bash">
{"arguments": {"command": "find src -type f | head -5", "timeout": 10}}
</tool_call>
</tool_call>`;

    const result = parseToolCalls(sample, "seed");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual([
      "read",
      "read",
      "read",
      "bash",
    ]);
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      path: "/tmp/README.md",
      limit: 100,
    });
    expect(JSON.parse(result.toolCalls[2]?.function.arguments ?? "{}")).toEqual({
      path: "/tmp/package.json",
    });
    expect(JSON.parse(result.toolCalls[3]?.function.arguments ?? "{}")).toEqual({
      command: "find src -type f | head -5",
      timeout: 10,
    });
    expect(result.content).toBe("");
  });

  it("parses name-first bare JSON and preserves canonical arguments-name key order", () => {
    const bare = '{"name":"bash","arguments":{"timeout":10,"command":"pwd"}}';
    const result = parseToolCalls(bare, "seed");

    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function).toEqual({
      name: "bash",
      arguments: '{"command":"pwd","timeout":10}',
    });
    expect(canonicalAssistantText(bare)).toBe(
      '<tool_call>\n{"arguments":{"command":"pwd","timeout":10},"name":"bash"}\n</tool_call>',
    );
  });

  it("accepts mangled tags and bare JSON tool payloads", () => {
    const mangled = parseToolCalls(
      '<_call>\n{"name":"list_files","arguments":{"path":"."}}\n</tool_call>',
      "seed",
    );
    expect(mangled.toolCalls).toHaveLength(1);
    expect(mangled.toolCalls[0]?.function).toEqual({
      name: "list_files",
      arguments: '{"path":"."}',
    });

    const bare = parseToolCalls('{"name":"read_file","arguments":{"path":"README.md"}}', "seed");
    expect(bare.toolCalls[0]?.function.name).toBe("read_file");
  });

  it("harvests args-before-name bare JSON alongside tagged calls", () => {
    const result = parseToolCalls(`<tool_call>
{"name":"read","arguments":{"path":"README.md"}}
</tool_call>
{"arguments":{"path":"package.json"},"name":"read"}`);

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function)).toEqual([
      { name: "read", arguments: '{"path":"README.md"}' },
      { name: "read", arguments: '{"path":"package.json"}' },
    ]);
  });

  it("deduplicates identical tagged and bare calls while stripping both", () => {
    const payload = '{"arguments":{"path":"package.json"},"name":"read"}';
    const result = parseToolCalls(`<tool_call>${payload}</tool_call>\n${payload}`);

    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function).toEqual({
      name: "read",
      arguments: '{"path":"package.json"}',
    });
  });

  it("repairs both truncated bare reads from failed session A", () => {
    const result = parseToolCalls(FAILED_SESSION_A, "session-a");

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["read", "read"]);
    expect(result.toolCalls.map((call) => JSON.parse(call.function.arguments))).toEqual([
      {
        path: "/Users/kittors/Developer/opensource/deepseek-web-api/.github/workflows/ci.yml",
      },
      { path: "/Users/kittors/Developer/opensource/deepseek-web-api/tsconfig.json" },
    ]);
  });

  it("repairs the cross-closed truncated bash from failed session B", () => {
    const result = parseToolCalls(FAILED_SESSION_B, "session-b");

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["bash"]);
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      command: "find /Users/kittors/Developer/opensource/deepseek-web-api/src -name '*.ts' | sort",
      timeout: 5,
    });
    expect(canonicalParsedAssistantText(result)).toBe(
      '<tool_call>\n{"arguments":{"command":"find /Users/kittors/Developer/opensource/deepseek-web-api/src -name \'*.ts\' | sort","timeout":5},"name":"bash"}\n</tool_call>',
    );
  });

  it.each([
    ["<_call>", "</tool_call>"],
    ["<_call>", "</_call>"],
    ["<tool_call>", "</_call>"],
    ["<tool_call>", "</tool_call>"],
  ])("repairs loose tag pair %s ... %s", (open, close) => {
    const result = parseToolCalls(
      `${open}\n{"name":"read","arguments":{"path":"README.md"}\n${close}`,
    );

    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function).toEqual({
      name: "read",
      arguments: '{"path":"README.md"}',
    });
  });

  it("repairs all three truncated reads from failed session C", () => {
    const result = parseToolCalls(FAILED_SESSION_C, "session-c");

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["read", "read", "read"]);
    expect(result.toolCalls.map((call) => JSON.parse(call.function.arguments).path)).toEqual([
      "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/client.ts",
      "/Users/kittors/Developer/opensource/deepseek-web-api/src/server/routes.ts",
      "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/completion.ts",
    ]);
  });

  it("repairs several missing nested closers without changing their data", () => {
    const result = parseToolCalls(
      '{"name":"write","arguments":{"config":{"items":[{"path":"a"},{"path":"b"}]',
    );

    expect(result.content).toBe("");
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      config: { items: [{ path: "a" }, { path: "b" }] },
    });
  });

  it("harvests a complete tagged call beside a truncated bare call", () => {
    const result = parseToolCalls(`<tool_call>
{"name":"bash","arguments":{"command":"pwd"}}
</tool_call>
{"arguments":{"path":"package.json"},"name":"read"`);

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function)).toEqual([
      { name: "bash", arguments: '{"command":"pwd"}' },
      { name: "read", arguments: '{"path":"package.json"}' },
    ]);
  });

  it("keeps a valid bash call while repairing both failed session A reads", () => {
    const result = parseToolCalls(`<tool_call>
{"name":"bash","arguments":{"command":"ls -la /Users/kittors/Developer/opensource/deepseek-web-api/src/","timeout":5}}
</tool_call>
${FAILED_SESSION_A}`);

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["bash", "read", "read"]);
  });

  it("repairs an unclosed tagged block at end of output", () => {
    const result = parseToolCalls('<tool_call>\n{"name":"read","arguments":{"path":"README.md"');

    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function).toEqual({
      name: "read",
      arguments: '{"path":"README.md"}',
    });
  });

  it("does not swallow prose or fenced JSON documentation examples", () => {
    const prose = 'Example payload:\n{"name":"read","arguments":{"path":"README.md"}}';
    const fenced = '```json\n{"name":"read","arguments":{"path":"README.md"}}\n```';

    expect(parseToolCalls(prose)).toEqual({ content: prose, toolCalls: [] });
    expect(parseToolCalls(fenced)).toEqual({ content: fenced, toolCalls: [] });
  });

  it("cleans protocol-only garbage when an unfinished string cannot be repaired safely", () => {
    const garbage = '<_call>\n{"name":"read","arguments":{"path":"README.md}\n</tool_call>';

    expect(parseToolCalls(garbage)).toEqual({ content: "", toolCalls: [] });
  });

  it("promotes tool tags leaked into reasoning when output has none", () => {
    const result = parseToolCallsFromParts(
      "",
      'I should list files.\n<tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>',
      "seed",
    );
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function).toEqual({
      name: "bash",
      arguments: '{"command":"ls"}',
    });
    expect(canonicalParsedAssistantText(result)).not.toContain("I should list files.");
  });

  it("prefers output tool calls over reasoning ones", () => {
    const result = parseToolCallsFromParts(
      '<tool_call>\n{"name":"read","arguments":{"path":"a"}}\n</tool_call>',
      '<tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>',
      "seed",
    );
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("read");
  });

  it("promotes a truncated call found only in reasoning", () => {
    const result = parseToolCallsFromParts("", FAILED_SESSION_B, "reasoning-repair");

    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("bash");
  });

  it("returns cleaned reasoning when no RESPONSE text exists", () => {
    expect(parseToolCallsFromParts("", "A normal final answer.")).toEqual({
      content: "A normal final answer.",
      toolCalls: [],
    });
    expect(
      parseToolCallsFromParts(
        "",
        '<_call>\n{"name":"read","arguments":{"path":"README.md}\n</tool_call>',
      ),
    ).toEqual({ content: "", toolCalls: [] });
  });
});

describe("parseToolCalls DSML native protocol", () => {
  const webRunCode = `<｜｜DSML｜｜calls>
<｜｜DSML｜｜ invoke name="run_code">
<｜｜DSML｜｜ parameter name="code" string="true">const top = await tools.pwsh({ command: "Get-ChildItem -Force -Name" });
console.log(top);</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="description" string="true">Inspect backend project layout</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>`;

  it("parses the double-bar spaced web stanza (run_code) into a function call", () => {
    const result = parseToolCalls(webRunCode, "seed");

    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("run_code");
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      code: 'const top = await tools.pwsh({ command: "Get-ChildItem -Force -Name" });\nconsole.log(top);',
      description: "Inspect backend project layout",
    });
  });

  it("parses single-bar tool_calls with typed string and JSON parameters", () => {
    const stanza = [
      "<｜DSML｜tool_calls>",
      '<｜DSML｜invoke name="set_count">',
      '<｜DSML｜parameter name="count" string="false">42</｜DSML｜parameter>',
      '<｜DSML｜parameter name="enabled" string="false">true</｜DSML｜parameter>',
      '<｜DSML｜parameter name="label" string="true">ok</｜DSML｜parameter>',
      '<｜DSML｜parameter name="options" string="false">{"nested":[1,2]}</｜DSML｜parameter>',
      "</｜DSML｜invoke>",
      "</｜DSML｜tool_calls>",
    ].join("");

    const result = parseToolCalls(stanza, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      count: 42,
      enabled: true,
      label: "ok",
      options: { nested: [1, 2] },
    });
  });

  it("keeps prose around a DSML stanza and emits multiple invokes in order", () => {
    const stanza = `Running it now.
<｜｜DSML｜｜calls>
<｜｜DSML｜｜invoke name="bash"><｜｜DSML｜｜parameter name="command" string="true">ls</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>
<｜｜DSML｜｜invoke name="read"><｜｜DSML｜｜parameter name="path" string="true">README.md</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>
</｜｜DSML｜｜calls>
Done setup.`;

    const result = parseToolCalls(stanza, "seed");
    expect(result.content).toBe("Running it now.\n\nDone setup.");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["bash", "read"]);
  });

  it("unescapes closing parameter tags inside string bodies", () => {
    const stanza =
      '<｜｜DSML｜｜invoke name="run_code">' +
      '<｜｜DSML｜｜parameter name="code" string="true">' +
      "a&lt;/｜｜DSML｜｜parameter>b&amp;lt;/｜｜DSML｜｜parameter>c" +
      "</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>";

    const result = parseToolCalls(stanza, "seed");
    expect(result.content).toBe("");
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      code: "a</｜｜DSML｜｜parameter>b&lt;/｜｜DSML｜｜parameter>c",
    });
  });

  it("does not harvest JSON inside DSML parameters as extra bare calls", () => {
    const stanza = `<｜DSML｜tool_calls>
<｜DSML｜invoke name="read">
<｜DSML｜parameter name="query" string="false">
{"name":"bash","arguments":{"command":"ls"}}
</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>`;

    const result = parseToolCalls(stanza, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["read"]);
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      query: { name: "bash", arguments: { command: "ls" } },
    });
  });

  it("repairs an unclosed DSML stanza truncated at end of output", () => {
    const result = parseToolCalls(
      '<｜｜DSML｜｜invoke name="read"><｜｜DSML｜｜parameter name="path" string="true">README.md',
      "seed",
    );
    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function).toEqual({
      name: "read",
      arguments: '{"path":"README.md"}',
    });
  });

  it("strips nameless DSML protocol garbage while preserving surrounding prose", () => {
    const stanza =
      "Before.\n" +
      '<｜｜DSML｜｜invoke><｜｜DSML｜｜parameter name="x" string="true">y</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>';

    expect(parseToolCalls(stanza, "seed")).toEqual({ content: "Before.", toolCalls: [] });
    expect(parseToolCalls(stanza.replace("Before.\n", ""), "seed")).toEqual({
      content: "",
      toolCalls: [],
    });
  });

  it("promotes DSML calls leaked into reasoning and canonicalizes to <tool_call>", () => {
    const result = parseToolCallsFromParts("", webRunCode, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("run_code");
    expect(canonicalParsedAssistantText(result)).toBe(
      '<tool_call>\n{"arguments":{"code":"const top = await tools.pwsh({ command: \\"Get-ChildItem -Force -Name\\" });\\nconsole.log(top);","description":"Inspect backend project layout"},"name":"run_code"}\n</tool_call>',
    );
  });

  it("deduplicates the same call emitted once as DSML and once as <tool_call>", () => {
    const mixed =
      webRunCode +
      '\n<tool_call>\n{"name":"run_code","arguments":{"description":"Inspect backend project layout","code":"const top = await tools.pwsh({ command: \\"Get-ChildItem -Force -Name\\" });\\nconsole.log(top);"}}\n</tool_call>';

    const result = parseToolCalls(mixed, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("run_code");
  });

  it("strips the mangled ASCII < calls> wrapper seen in real web output", () => {
    const real = `< calls> <｜｜DSML｜｜ invoke name="run_code"> <｜｜DSML｜｜ parameter name="code" string="true">const x = 1;</｜｜DSML｜｜ parameter> <｜｜DSML｜｜ parameter name="description" string="true">List backend tree</｜｜DSML｜｜ parameter> </｜｜DSML｜｜ invoke> </｜｜DSML｜｜ calls>`;
    const result = parseToolCalls(real, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("run_code");
    expect(canonicalAssistantText(real)).not.toContain("< calls>");
    expect(canonicalAssistantText(real)).toContain("<tool_call>");
  });

  it("treats a lone residue wrapper as protocol-only output", () => {
    expect(parseToolCalls("< calls>", "seed")).toEqual({ content: "", toolCalls: [] });
    expect(parseToolCalls("  </tool_calls> ", "seed")).toEqual({ content: "", toolCalls: [] });
  });

  it("strips residue prefixing an ordinary <tool_call> block", () => {
    const text =
      '< calls> <tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>';
    const result = parseToolCalls(text, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function.name).toBe("bash");
  });

  it("keeps prose that merely mentions a calls-like token", () => {
    const text = "The function calls> helper is unrelated to tooling.";
    expect(parseToolCalls(text, "seed")).toEqual({ content: text, toolCalls: [] });
  });

  it("recovers an unwrapped shell command object without name or arguments keys", () => {
    const body = '{"command":"npx --yes typescript@5.4.5 tsc --noEmit","blocking":true,' +
      '"command_type":"short_running_process",' +
      '"cwd":"f:\\\\workspace\\\\play-together\\\\miniprogram","requires_approval":false}';
    const closed = `<tool_call>\n${body}\n</tool_call>`;
    const result = parseToolCalls(closed, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("bash");
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      command: "npx --yes typescript@5.4.5 tsc --noEmit",
      cwd: "f:\\workspace\\play-together\\miniprogram",
    });

    // The same object truncated without a close tag is recovered at flush.
    const unclosed = `<tool_call>\n${body}`;
    const repaired = parseToolCalls(unclosed, "seed");
    expect(repaired.content).toBe("");
    expect(repaired.toolCalls[0]?.function.name).toBe("bash");
  });

  it("does not treat a command-less object as a shell call", () => {
    const text = '<tool_call>\n{"city":"Hefei"}\n</tool_call>';
    const result = parseToolCalls(text, "seed");
    expect(result.toolCalls).toEqual([]);
  });

  it("infers an omitted name from registered parameter keys", () => {
    const body = String.raw`{"arguments":{"pattern":"Options<","path":"f:\workspace\x","glob":"*.d.ts","output_mode":"content","-n":true,"head_limit":40}}`;
    const sample = `<_call>\n${body}\n</tool_call>\n</｜｜DSML｜｜ calls>`;
    const tools = [
      { name: "Glob", paramKeys: ["pattern", "path", "glob"] },
      { name: "Grep", paramKeys: ["pattern", "path", "glob", "output_mode", "-n", "head_limit"] },
      { name: "Read", paramKeys: ["file_path"] },
    ];
    const result = parseToolCalls(sample, "seed", { tools });
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("Grep");
    const args = JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}") as { path: string };
    expect(args.path).toBe(String.raw`f:\workspace\x`);
  });

  it("does not infer a name when two tools explain every key", () => {
    const text = '<tool_call>\n{"arguments":{"pattern":"x"}}\n</tool_call>';
    const tools = [
      { name: "A", paramKeys: ["pattern"] },
      { name: "B", paramKeys: ["pattern"] },
    ];
    expect(parseToolCalls(text, "seed", { tools }).toolCalls).toEqual([]);
  });

  it("recovers a call behind an empty <> opener and an orphan close", () => {
    const body = '{"name":"TodoWrite","arguments":{"merge":true,"todos":[]}}';
    const result = parseToolCalls(`<>\n${body}\n</call>`, "seed", {
      tools: [{ name: "TodoWrite", paramKeys: ["todos", "merge"] }],
    });
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function).toEqual({
      name: "TodoWrite",
      arguments: '{"merge":true,"todos":[]}',
    });
  });

  it("does not treat <> inside ordinary prose as an opener", () => {
    const text = "SQL 里用 a <> b 做不等值比较。";
    expect(parseToolCalls(text, "seed")).toEqual({ content: text, toolCalls: [] });
  });

  it("repairs unescaped double quotes inside a command string", () => {
    const body = '{"name":"RunCommand","arguments":{"command":"python -c "print(\'ok\')"","blocking":true,' +
      '"cwd":"f:\\workspace\\play-together"}}';
    const result = parseToolCalls(`<tool_call>\n${body}\n</tool_call>`, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("RunCommand");
    const args = JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}") as {
      command: string; blocking: boolean; cwd: string;
    };
    expect(args.command).toBe("python -c \"print('ok')\"");
    expect(args.blocking).toBe(true);
    expect(args.cwd).toBe("f:\\workspace\\play-together");
  });

  it("does not greedily accept non-JSON or unclosed objects", () => {
    expect(parseToolCalls("<tool_call>{not json}</tool_call>")).toEqual({
      content: "<tool_call>{not json}</tool_call>",
      toolCalls: [],
    });
    const unclosed = '<_call>\n{"name":"read","arguments":{"path":"README.md}\n</tool_call>';
    expect(parseToolCalls(unclosed, "seed")).toEqual({ content: "", toolCalls: [] });
  });

  it("keeps separate keys when a string value embeds triple-quoted source", () => {
    const embedded =
      '"""主题包 + Planner 单元测试。\n' +
      'THEME_FIELDS = {\n"theme_key", "festival", "title",\n}\n' +
      'payload = json.loads(prompt.split("\\n", 1)[1])\n' +
      'assert "base_missions" not in payload, "prompt 不能携带成稿任务"\n"';
    const body = '{"name":"Write","arguments":{"file_path":"f:\\workspace\\play-together\\backend\\tests\\t.py",' +
      `"content":"${embedded}"}}`;
    const result = parseToolCalls(`<tool_call>\n${body}\n</tool_call>`, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("Write");
    const args = JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}") as {
      file_path: string; content: string;
    };
    expect(Object.keys(args).sort()).toEqual(["content", "file_path"]);
    expect(args.file_path).toBe("f:\\workspace\\play-together\\backend\\tests\\t.py");
    expect(args.content).toBe(embedded);
  });

  it("recovers orphan parameters missing invoke openers or shifted close tags", () => {
    const base = "f:\\workspace\\play-together\\miniprogram\\pages\\";
    const orphanInvokeClose = (file: string): string =>
      `<｜｜DSML｜｜ parameter name="file_path" string="true">${base}${file}</｜｜DSML｜｜ invoke>`;
    const orphanParameter = (file: string): string =>
      `<｜｜DSML｜｜ parameter name="file_path" string="true">${base}${file}</｜｜DSML｜｜ parameter>`;
    const stanza = [
      "<｜｜DSML｜｜ calls>",
      '<｜｜DSML｜｜ invoke name="Read">',
      orphanParameter("home\\home.ts"),
      "</｜｜DSML｜｜ invoke>",
      orphanInvokeClose("setup\\setup.ts"),
      orphanInvokeClose("setup\\setup.wxml"),
      orphanParameter("activity\\activity.ts"),
      "</｜｜DSML｜｜ parameter>",
      orphanParameter("activity\\activity.wxml"),
      "</｜｜DSML｜｜ parameter>",
      '<｜｜DSML｜｜ invoke name="Read">',
      orphanParameter("session\\session.ts"),
      "</｜｜DSML｜｜ parameter>",
      "</｜｜DSML｜｜ invoke>",
      orphanInvokeClose("session\\session.wxml"),
      '<｜｜DSML｜｜ invoke name="Read">',
      orphanParameter("complete\\complete.ts"),
      "</｜｜DSML｜｜ parameter>",
      "</｜｜DSML｜｜ invoke>",
      orphanInvokeClose("complete\\complete.wxml"),
      '<｜｜DSML｜｜ invoke name="Read">',
      orphanParameter("memory\\memory.ts"),
      "</｜｜DSML｜｜ parameter>",
      "</｜｜DSML｜｜ invoke>",
      orphanInvokeClose("memory\\memory.wxml"),
      '<｜｜DSML｜｜ invoke name="Read">',
      orphanParameter("memories\\memories.ts"),
      "</｜｜DSML｜｜ parameter>",
      "</｜｜DSML｜｜ invoke>",
      orphanInvokeClose("memories\\memories.wxml"),
      "</｜｜DSML｜｜ calls>",
    ].join("\n");

    const result = parseToolCalls(stanza, "seed");
    expect(result.content).toBe("");
    const files = result.toolCalls.map((call) => {
      expect(call.function.name).toBe("Read");
      return JSON.parse(call.function.arguments).file_path;
    });
    expect(files).toEqual([
      `${base}home\\home.ts`,
      `${base}setup\\setup.ts`,
      `${base}setup\\setup.wxml`,
      `${base}activity\\activity.ts`,
      `${base}activity\\activity.wxml`,
      `${base}session\\session.ts`,
      `${base}session\\session.wxml`,
      `${base}complete\\complete.ts`,
      `${base}complete\\complete.wxml`,
      `${base}memory\\memory.ts`,
      `${base}memory\\memory.wxml`,
      `${base}memories\\memories.ts`,
      `${base}memories\\memories.wxml`,
    ]);
  });

  it.each([
    ["NBSP", " "],
    ["ideographic space", "　"],
  ])("tolerates %s between DSML bars and the keyword", (_label, gap) => {
    const stanza = [
      `<｜｜DSML｜｜${gap}calls>`,
      `<｜｜DSML｜｜${gap}invoke name="LS">`,
      `<｜｜DSML｜｜${gap}parameter name="path" string="true">src</｜｜DSML｜｜parameter>`,
      `</｜｜DSML｜｜${gap}invoke>`,
      `</｜｜DSML｜｜${gap}calls>`,
    ].join("\n");
    const result = parseToolCalls(stanza, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]!.function.name).toBe("LS");
    expect(JSON.parse(result.toolCalls[0]!.function.arguments)).toEqual({ path: "src" });
  });

  it("parses two adjacent DSML wrappers with only whitespace between them", () => {
    const one = (name: string, argName: string, value: string) => [
      "<｜｜DSML｜｜ calls>",
      `<｜｜DSML｜｜ invoke name="${name}">`,
      `<｜｜DSML｜｜ parameter name="${argName}" string="true">${value}</｜｜DSML｜｜ parameter>`,
      "</｜｜DSML｜｜ invoke>",
      "</｜｜DSML｜｜ calls>",
    ].join("\n");
    const result = parseToolCalls(`${one("LS", "path", "src")}    ${one("Read", "file", "a.ts")}`, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["LS", "Read"]);
  });
});

describe("harness tool-result protocol", () => {
  const block = `<tool_call_result>
<toolcall_status>Done</toolcall_status>
<command_id>job-7c9a</command_id>
<command_status>Exited</command_status>
<command_run_logs>81 passed in 2.51s</command_run_logs>
</tool_call_result>`;

  it("hides a complete tool-result block as protocol-only", () => {
    expect(parseToolCalls(block)).toEqual({ content: "", toolCalls: [] });
  });

  it("hides a result block whose close tag never arrived", () => {
    expect(parseToolCalls(block.replace("</tool_call_result>", ""))).toEqual({ content: "", toolCalls: [] });
  });

  it("hides stranded inner harness tags without a wrapper", () => {
    const stray = "<command_id>job-1</command_id><command_status>Exited</command_status>";
    expect(parseToolCalls(stray)).toEqual({ content: "", toolCalls: [] });
  });

  it("keeps prose that precedes the result block", () => {
    expect(parseToolCalls(`Done.\n${block}`).content).toBe("Done.");
  });

  it("does not treat a bare result element as protocol", () => {
    const text = "<result>42</result>";
    expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
  });
});

describe("tool key and malformed DSML recovery", () => {
  const winPath = String.raw`d:\tmp\tijian`;
  const lsBlock = `<_call>\n{"tool": "LS", "arguments": {"path": "${winPath}"}}\n</_call>`;
  const grepWrapper = `<｜｜DSML｜｜ calls>\n` +
    `{"tool": "Grep", "arguments": {"pattern": "M6", "path": "${winPath}", ` +
    `"output_mode": "files_with_matches"}}\n` +
    `</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>`;

  it("recovers a call that names the tool with the tool key", () => {
    const result = parseToolCalls(lsBlock);
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("LS");
    expect(result.toolCalls[0]?.function.arguments).toBe(
      JSON.stringify({ path: winPath }),
    );
  });

  it("recovers bare JSON from a malformed DSML wrapper with shifted closes", () => {
    const result = parseToolCalls(grepWrapper);
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("Grep");
    expect(JSON.parse(result.toolCalls[0]!.function.arguments)).toMatchObject({
      pattern: "M6", output_mode: "files_with_matches",
    });
  });

  it("recovers both calls from the combined malformed turn", () => {
    const result = parseToolCalls(`${lsBlock}\n${grepWrapper}`);
    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["LS", "Grep"]);
  });

  it("keeps a tab escape inside a string without a drive prefix", () => {
    const result = parseToolCalls('<tool_call>{"name":"echo","arguments":{"text":"a\\tb"}}</tool_call>');
    expect(result.toolCalls[0]?.function.arguments).toBe(JSON.stringify({ text: "a\tb" }));
  });
});

const PARALLEL_OBJECTS = [
  '{"name":"read","arguments":{"path":"a.ts"}}',
  '{"name":"bash","arguments":{"command":"ls"}}',
  '{"name":"Grep","arguments":{"pattern":"x"}}',
];

describe("parseToolCalls multiple same-line bare objects", () => {
  it.each([
    ["space", " "],
    ["comma", ","],
    ["comma-space", ", "],
  ])("recovers three %s-separated bare objects", (_label, sep) => {
    const result = parseToolCalls(PARALLEL_OBJECTS.join(sep), "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function)).toEqual([
      { name: "read", arguments: '{"path":"a.ts"}' },
      { name: "bash", arguments: '{"command":"ls"}' },
      { name: "Grep", arguments: '{"pattern":"x"}' },
    ]);
  });

  it("recovers space-separated objects inside one tool-call tag", () => {
    const result = parseToolCalls(`<tool_call>${PARALLEL_OBJECTS.join(" ")}</tool_call>`, "seed");
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(3);
  });

  it("does not bridge a second object across prose on the same line", () => {
    const text = `${PARALLEL_OBJECTS[0]} then ${PARALLEL_OBJECTS[1]}`;
    const result = parseToolCalls(text, "seed");
    expect(result.toolCalls).toEqual([]);
    expect(result.content).toBe(text);
  });

  it("does not treat prose followed by a comma and an object as a bare turn", () => {
    const text = `see, ${PARALLEL_OBJECTS[0]}`;
    const result = parseToolCalls(text, "seed");
    expect(result.toolCalls).toEqual([]);
    expect(result.content).toBe(text);
  });
});

describe("parseToolCalls plain-XML invoke protocol", () => {
  const open = (tag: string, attrs = ""): string => "<" + tag + attrs + ">";
  const close = (tag: string): string => "</" + tag + ">";
  const param = (name: string, body: string, attrs = ""): string =>
    open("parameter", ` name="${name}"${attrs}`) + body + close("parameter");
  const wrapped = (body: string): string => open("call") + " " + body + " " + close("call");

  it("recovers the user's call/invoke/parameter stanza with raw string values", () => {
    const filePath = "c:\\Users\\研发部\\.trae-cn\\plugins\\lark\\1.0.5\\lark-base-workflow-schema.md";
    const body = open("invoke", ' name="Read"') + " " + param("file_path", filePath) + close("invoke");
    const result = parseToolCalls(wrapped(body));
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("Read");
    expect(result.toolCalls[0]?.function.arguments).toBe(JSON.stringify({ file_path: filePath }));
  });

  it("parses a standalone invoke without a call wrapper, honoring typed values", () => {
    const typed =
      param("line", "42", ' string="false"') + param("tags", '["a","b"]') + param("note", "x &amp; y", ' string="true"');
    const result = parseToolCalls(open("invoke", ' name="Edit"') + typed + close("invoke"));
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.arguments).toBe(
      JSON.stringify({ line: 42, note: "x & y", tags: ["a", "b"] }),
    );
  });

  it("recovers an unclosed invoke at turn end", () => {
    const text = open("call") + open("invoke", ' name="Read"') + param("file", "f:\\workspace\\x.md");
    const result = parseToolCalls(text);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("Read");
    expect(result.content).toBe("");
  });

  it("does not fabricate calls from nameless invokes or stranded parameters", () => {
    expect(parseToolCalls(open("invoke") + param("x", "v") + close("invoke")).toolCalls).toEqual([]);
    const stranded = param("x", "v");
    expect(parseToolCalls(stranded)).toEqual({ content: stranded, toolCalls: [] });
  });

  it("leaves ordinary prose untouched", () => {
    const text = "please call support, numbers like a < b stay prose";
    expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
  });
});
