/** Verifies the documented behavior of the corresponding production module. */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { SessionStore } from "../../src/deepseek/sessionStore.js";
import { canonicalAssistantText } from "../../src/deepseek/toolCalls.js";
import type { MessageTurn } from "../../src/deepseek/types.js";

function remember(
  store: SessionStore,
  sessionId: string,
  fullTurns: MessageTurn[],
  assistantContent: string,
  extras: { modelType?: "default" | "expert"; responseMessageId?: MessageIdLike; convKey?: string } = {},
): void {
  store.remember({
    sessionId,
    modelType: extras.modelType ?? "default",
    responseMessageId: extras.responseMessageId ?? 1,
    ...(extras.convKey ? { convKey: extras.convKey } : {}),
    fullTurns,
    assistantContent,
  });
}

type MessageIdLike = string | number;

const block = (name: string, args: unknown): string =>
  canonicalAssistantText(`<tool_call>\n{"name":"${name}","arguments":${JSON.stringify(args)}}\n</tool_call>`);

describe("SessionStore", () => {
  it("does not persist or advance a session for an empty assistant response", () => {
    const store = new SessionStore();
    remember(store, "empty-session", [{ role: "user", content: "inspect" }], "   ", {
      modelType: "expert", responseMessageId: 99,
    });

    expect(store.get("empty-session")).toBeUndefined();
    expect(store.resolve({ conversation: "empty-session" })).toMatchObject({
      sessionId: "empty-session",
      parentMessageId: null,
      createIfMissing: true,
    });
  });

  it("matches full history across model switches", () => {
    const store = new SessionStore();
    remember(store, "session-1", [{ role: "user", content: "hello" }], "world", {
      responseMessageId: 42,
    });

    const resolution = store.resolve({
      model: "deepseek-v4-pro",
      messages: [
        { role: "system", content: "Keep the same conversation." },
        { role: "user", content: "hello" },
        { role: "assistant", content: "world" },
        { role: "user", content: "continue" },
      ],
    });

    expect(resolution).toMatchObject({ sessionId: "session-1", parentMessageId: 42 });
  });

  it("reuses a long Pi history after the persisted turn window is truncated", () => {
    const store = new SessionStore();
    const messages: Array<Record<string, unknown>> = [];
    for (let index = 0; index < 22; index += 1) {
      const user = { role: "user", content: `request-${index}` };
      const assistant = { role: "assistant", content: `answer-${index}` };
      messages.push(user, assistant);
      remember(
        store,
        "long-session",
        messages.slice(0, -1).map((turn) => ({ role: String(turn.role), content: String(turn.content) })),
        `answer-${index}`,
        { modelType: "expert", responseMessageId: index + 1 },
      );
    }

    expect(store.get("long-session")?.turns).toHaveLength(40);
    expect(store.resolve({ messages: [...messages, { role: "user", content: "continue" }] }))
      .toMatchObject({ sessionId: "long-session", parentMessageId: 22 });
  });

  it("resolves previous_response_id", () => {
    const store = new SessionStore();
    const sessionId = "123e4567-e89b-12d3-a456-426614174000";
    remember(store, sessionId, [{ role: "user", content: "a" }], "b", {
      modelType: "expert", responseMessageId: 7,
    });
    expect(store.resolve({ previous_response_id: `resp_${sessionId}` })).toMatchObject({
      sessionId,
      parentMessageId: 7,
    });
  });

  it("does not include model type in fingerprint keys", () => {
    const store = new SessionStore();
    remember(store, "same-session", [{ role: "user", content: "hello" }], "world", {
      convKey: "fp:user:hello\n---\nassistant:world",
    });
    expect(
      store.resolve({
        model: "pro",
        input: [
          { role: "user", content: "hello" },
          { role: "assistant", content: "world" },
          { role: "user", content: "next" },
        ],
      }).sessionId,
    ).toBe("same-session");
  });

  it("matches a tool result against the preceding structured assistant call", () => {
    const store = new SessionStore();
    const callBlock = block("get_weather", { city: "Hefei" });
    remember(store, "tool-session", [{ role: "user", content: "weather" }], callBlock, {
      responseMessageId: 9,
    });

    expect(
      store.resolve({
        messages: [
          { role: "user", content: "weather" },
          {
            role: "assistant",
            tool_calls: [{ function: { name: "get_weather", arguments: '{"city":"Hefei"}' } }],
          },
          { role: "tool", content: '{"temperature":32}' },
        ],
      }),
    ).toMatchObject({ sessionId: "tool-session", parentMessageId: 9 });
  });

  it("reuses structured tool history while storing no reasoning prose", () => {
    const store = new SessionStore();
    const callBlock = block("bash", { command: "date" });
    remember(store, "hidden-tool-session", [{ role: "user", content: "今天日期" }], callBlock, {
      modelType: "expert", responseMessageId: 12,
    });

    expect(callBlock).toBe(
      '<tool_call>\n{"arguments":{"command":"date"},"name":"bash"}\n</tool_call>',
    );
    expect(callBlock).not.toContain("Need current data");
    expect(store.get("hidden-tool-session")?.turns.at(-1)?.content).toBe(
      '<tool_call> {"arguments":{"command":"date"},"name":"bash"} </tool_call>',
    );
    expect(
      store.resolve({
        messages: [
          { role: "user", content: "今天日期" },
          {
            role: "assistant",
            tool_calls: [{ function: { name: "bash", arguments: '{"command":"date"}' } }],
          },
          { role: "tool", tool_call_id: "call_date", content: "Sat Jul 25" },
        ],
      }),
    ).toMatchObject({ sessionId: "hidden-tool-session", parentMessageId: 12 });
  });

  it("keeps an agent tool-call turn and its result in one session", () => {
    const store = new SessionStore();
    const callBlock = block("Read", { file_path: "f:/x.ts" });
    // Round 1: user asks, model emits a tool call.
    remember(store, "agent-session", [{ role: "user", content: "read this" }], callBlock, {
      responseMessageId: 10,
    });

    // Round 2: client replays user/assistant call, adds the tool result.
    const round2: MessageTurn[] = [
      { role: "user", content: "read this" },
      { role: "assistant", content: callBlock },
      { role: "tool", content: "export const value = 1" },
    ];
    const resolution = store.resolve({ messages: round2 });
    expect(resolution.sessionId).toBe("agent-session");
    remember(store, "agent-session", round2, "The file exports value 1.", {
      responseMessageId: 11,
    });

    // Stored turns keep the expanded form: user, assistant call, tool, answer.
    const turns = store.get("agent-session")?.turns ?? [];
    expect(turns.map((turn) => turn.role)).toEqual([
      "user", "assistant", "tool", "assistant",
    ]);

    // Round 3: a follow-up question still resolves the same session.
    const followUp = store.resolve({
      messages: [
        ...turns,
        { role: "user", content: "what is exported?" },
      ],
    });
    expect(followUp.sessionId).toBe("agent-session");
    expect(followUp.parentMessageId).toBe(11);
  });

  it("upgrades a legacy canonical session to expanded form on the next turn", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ds-sessions-"));
    const file = path.join(dir, "sessions.json");
    const callBlock = block("Read", { file_path: "f:/x.ts" });
    fs.writeFileSync(
      file,
      JSON.stringify({
        // Legacy storage collapsed the tool turn into one canonical assistant.
        sessions: {
          "legacy-agent": {
            lastResponseMessageId: 20,
            updatedAt: Date.now(),
            turns: [
              { role: "user", content: "read this" },
              { role: "assistant", content: callBlock },
            ],
          },
        },
        convs: {},
      }),
    );
    const store = new SessionStore(file);

    // Client replay with expanded tool result still identifies the old session.
    const replay: MessageTurn[] = [
      { role: "user", content: "read this" },
      { role: "assistant", content: callBlock },
      { role: "tool", content: "const value = 1" },
    ];
    const resolution = store.resolve({ messages: replay });
    expect(resolution.sessionId).toBe("legacy-agent");
    remember(store, "legacy-agent", replay, "It exports value.", {
      responseMessageId: 21,
    });

    // The entry is upgraded: roles include the tool turn, no canonical collapse.
    const turns = store.get("legacy-agent")?.turns ?? [];
    expect(turns.map((turn) => turn.role)).toEqual([
      "user", "assistant", "tool", "assistant",
    ]);
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("continues a DSML thread when the client replays residue text and a function_call item", () => {
    const store = new SessionStore();
    const dsml =
      '< calls> <｜｜DSML｜｜invoke name="run_code"><｜｜DSML｜｜parameter name="code" string="true">ls</｜｜DSML｜｜parameter>' +
      '<｜｜DSML｜｜parameter name="description" string="true">List files</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>';
    remember(
      store,
      "dsml-session",
      [{ role: "user", content: "inspect backend" }],
      canonicalAssistantText(dsml),
      { responseMessageId: 5 },
    );

    const resolution = store.resolve({
      input: [
        { type: "message", role: "user", content: "inspect backend" },
        { type: "message", role: "assistant", content: "< calls>" },
        {
          type: "function_call",
          call_id: "call_x",
          name: "run_code",
          arguments: '{"code":"ls","description":"List files"}',
        },
        { type: "function_call_output", call_id: "call_x", output: "done" },
        { type: "message", role: "user", content: "继续" },
      ],
    });

    expect(resolution).toMatchObject({ sessionId: "dsml-session", parentMessageId: 5 });
  });

  it("migrates raw fingerprint keys and wrapper residue when loading from disk", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ds-sessions-"));
    const file = path.join(dir, "sessions.json");
    const legacyAssistant =
      "< calls> <tool_call> {\"arguments\":{\"city\":\"Hefei\"},\"name\":\"get_weather\"} </tool_call>";
    const rawFp = `user:weather\n---\nassistant:${legacyAssistant}`;
    fs.writeFileSync(
      file,
      JSON.stringify({
        sessions: {
          "legacy-session": {
            lastResponseMessageId: 7,
            updatedAt: Date.now(),
            turns: [
              { role: "user", content: "weather" },
              { role: "assistant", content: legacyAssistant },
              { role: "assistant", content: "< calls>" },
            ],
          },
        },
        convs: { [`fp:${rawFp}`]: "legacy-session" },
      }),
    );

    const store = new SessionStore(file);
    remember(store, "trigger-session", [{ role: "user", content: "ping" }], "pong");
    store.close();

    const onDisk = JSON.parse(fs.readFileSync(file, "utf8"));
    expect(Object.keys(onDisk.convs).every((key) => /^fp:[0-9a-f]{64}$/.test(key))).toBe(true);
    const assistantTurns = onDisk.sessions["legacy-session"].turns.filter(
      (turn: { role: string }) => turn.role === "assistant",
    );
    expect(assistantTurns).toHaveLength(1);
    expect(assistantTurns[0].content).not.toContain("< calls>");

    const reopened = new SessionStore(file);
    const resolution = reopened.resolve({
      messages: [
        { role: "user", content: "weather" },
        { role: "assistant", content: legacyAssistant },
        { role: "user", content: "next" },
      ],
    });
    expect(resolution).toMatchObject({ sessionId: "legacy-session", parentMessageId: 7 });

    fs.rmSync(dir, { recursive: true, force: true });
  });
});
