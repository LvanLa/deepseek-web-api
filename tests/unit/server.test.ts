/** Verifies the documented behavior of the corresponding production module. */
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";

import { ChromeManager } from "../../src/browser/chrome.js";
import { LoginManager } from "../../src/browser/login.js";
import type { AppConfig } from "../../src/config/env.js";
import { DeepSeekClient } from "../../src/deepseek/client.js";
import { SessionStore } from "../../src/deepseek/sessionStore.js";
import { createServer } from "../../src/server/createServer.js";
import { createLogger } from "../../src/utils/logger.js";
import type { RequestBody } from "../../src/deepseek/types.js";

const servers: Server[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
});

function testClient(): DeepSeekClient {
  const dataDir = mkdtempSync(path.join(tmpdir(), "deepseek-web-api-test-"));
  const config: AppConfig = {
    port: 8787,
    host: "127.0.0.1",
    cdpEndpoint: "http://127.0.0.1:9333",
    dataDir,
    authFile: path.join(dataDir, "auth.json"),
    apiKeyFile: path.join(dataDir, ".api-key"),
    sessionsFile: path.join(dataDir, "sessions.json"),
    chromeProfileDir: path.join(dataDir, "chrome-profile"),
    powWorkerUrl: "https://example.com/pow.js",
    baseUrl: "https://chat.deepseek.com",
    debug: false,
    toolReasoning: "hidden",
    showBrowser: false,
  };
  const logger = createLogger(false);
  const chrome = new ChromeManager(config, logger);
  const login = new LoginManager(chrome, config, logger);
  return new DeepSeekClient(config, login, new SessionStore(), logger);
}

async function baseUrl(server: Server): Promise<string> {
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  servers.push(server);
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("missing test server address");
  return `http://127.0.0.1:${address.port}`;
}

describe("HTTP server routes", () => {
  it("keeps health open and protects /v1 routes", async () => {
    const server = createServer({ client: testClient(), apiKeys: ["secret", "also-secret"], debug: false });
    const url = await baseUrl(server);
    expect(await fetch(`${url}/health`).then((response) => response.json())).toEqual({ ok: true });
    expect((await fetch(`${url}/v1/models`)).status).toBe(401);
    const models = await fetch(`${url}/v1/models`, {
      headers: { authorization: "Bearer secret" },
    });
    expect(models.status).toBe(200);
    expect(await models.json()).toMatchObject({ object: "list" });
    const alt = await fetch(`${url}/v1/models`, { headers: { "x-api-key": "also-secret" } });
    expect(alt.status).toBe(200);
  });
});

async function runRecoveryCase(
  stored: boolean, firstResponse: "empty" | "contextError" | "resultBlock" = "empty",
): Promise<Array<{ sessionId: unknown; parent: unknown }>> {
  const client = testClient();
  const bag = client as unknown as { login: Record<string, unknown>; sessions: SessionStore };
  if (stored) {
    bag.sessions.remember({
      sessionId: "00000000-0000-0000-0000-000000000001", modelType: "default",
      responseMessageId: 48, convKey: null, fullTurns: [], assistantContent: "prior answer",
      instructionFingerprint: "", toolsFingerprint: "",
    });
  }
  bag.login.dumpCurrent = async () => ({ token: "t", cookie: "", cookies: [], dumped_at: "" });
  bag.login.page = async () => ({
    evaluate: async (_script: unknown, arg: { arg: Record<string, unknown> }) => ({
      token: "t", powHeader: "p", modelType: "default",
      sessionId: arg.arg.sessionId ?? "00000000-0000-0000-0000-000000000002",
      reused: Boolean(arg.arg.reuseSession && arg.arg.sessionId),
    }),
  });

  const requested: Array<{ sessionId: unknown; parent: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (url: unknown, init?: { body?: string }) => {
    if (!String(url).includes("/completion"))
      return originalFetch(url as Parameters<typeof fetch>[0]);
    const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    requested.push({ sessionId: payload.chat_session_id, parent: payload.parent_message_id });
    if (requested.length === 1) {
      if (firstResponse === "contextError") {
        return new Response(
          'data: {"type":"error","content":"达到对话长度上限，请开启新对话",' +
          '"clear_response":true,"finish_reason":"context_length_exceeded"}\n\n',
        );
      }
      if (firstResponse === "resultBlock") {
        const block = "<tool_call_result><toolcall_status>Done</toolcall_status>" +
          "<command_id>job-x</command_id><command_run_logs>81 passed" +
          "</command_run_logs></tool_call_result>";
        const frame = JSON.stringify({
          p: "response/fragments", o: "APPEND",
          v: [{ type: "RESPONSE", content: block }],
        });
        return new Response(`data: ${frame}\n\nevent: close\ndata: {}\n\n`);
      }
      return new Response('event: close\ndata: {}\n\n');
    }
    return new Response(
      'data: {"p":"response/fragments","o":"APPEND","v":' +
      '[{"type":"RESPONSE","content":"salvaged answer"}]}\n\n' +
      'event: close\ndata: {}\n\n',
    );
  }) as typeof fetch;

  try {
    const body: RequestBody = {
      model: "deepseek-v4-flash",
      conversation: "00000000-0000-0000-0000-000000000001",
      parent_message_id: 48,
      messages: [{ role: "user", content: "fix it" }],
      tools: [{
        type: "function",
        function: { name: "run", parameters: { type: "object", properties: {} } },
      }],
    };
    const seen: string[] = [];
    await client.streamChat(body, (chunk) => {
      const delta = chunk.choices[0]?.delta as { content?: string } | undefined;
      if (typeof delta?.content === "string") seen.push(delta.content);
    });
    expect(seen.join("")).toBe("salvaged answer");
  } finally {
    globalThis.fetch = originalFetch;
  }
  return requested;
}

describe("silent empty tool-turn recovery", () => {
  it("retries in a fresh session when the reused upstream session was bloated", async () => {
    const requested = await runRecoveryCase(true);
    expect(requested).toEqual([
      { sessionId: "00000000-0000-0000-0000-000000000001", parent: 48 },
      { sessionId: "00000000-0000-0000-0000-000000000002", parent: null },
    ]);
  });

  it("stays in the current session when it was not a reused one", async () => {
    const requested = await runRecoveryCase(false);
    // An unknown explicit id creates a fresh session first; a non-reused turn
    // retries within that same session rather than forking again.
    expect(requested.map((entry) => entry.sessionId)).toEqual([
      "00000000-0000-0000-0000-000000000002",
      "00000000-0000-0000-0000-000000000002",
    ]);
    expect(requested[1]?.parent).toBeNull();
  });

  it("forks to a fresh session after a context length error and succeeds", async () => {
    const requested = await runRecoveryCase(true, "contextError");
    expect(requested).toEqual([
      { sessionId: "00000000-0000-0000-0000-000000000001", parent: 48 },
      { sessionId: "00000000-0000-0000-0000-000000000002", parent: null },
    ]);
  });

  it("retries in place after a context length error on a new session", async () => {
    const requested = await runRecoveryCase(false, "contextError");
    expect(requested.map((entry) => entry.sessionId)).toEqual([
      "00000000-0000-0000-0000-000000000002",
      "00000000-0000-0000-0000-000000000002",
    ]);
    expect(requested[1]?.parent).toBeNull();
  });

  it("hides an echoed tool-result block and retries in a fresh session", async () => {
    const requested = await runRecoveryCase(true, "resultBlock");
    expect(requested).toEqual([
      { sessionId: "00000000-0000-0000-0000-000000000001", parent: 48 },
      { sessionId: "00000000-0000-0000-0000-000000000002", parent: null },
    ]);
  });
});

async function runAuthRefreshCase(recover: boolean): Promise<{
  requested: Array<{ sessionId: unknown; parent: unknown }>; refreshes: number;
}> {
  const client = testClient();
  const bag = client as unknown as { login: Record<string, unknown>; sessions: SessionStore };
  bag.sessions.remember({
    sessionId: "00000000-0000-0000-0000-000000000001", modelType: "default",
    responseMessageId: 48, convKey: null, fullTurns: [], assistantContent: "prior answer",
    instructionFingerprint: "", toolsFingerprint: "",
  });
  let refreshes = 0, prepareAttempts = 0;
  bag.login.dumpCurrent = async () => ({ token: "t", cookie: "", cookies: [], dumped_at: "" });
  bag.login.refreshLogin = async () => { refreshes += 1; };
  bag.login.page = async () => ({
    evaluate: async () => {
      prepareAttempts += 1;
      if (prepareAttempts === 1 || !recover) {
        throw new Error('deepseek-auth-token-invalid: {"code":40003,' +
          '"msg":"Authorization Failed (invalid token)","data":null}');
      }
      return {
        token: "t2", powHeader: "p", modelType: "default",
        sessionId: "00000000-0000-0000-0000-000000000001", reused: true,
      };
    },
  });

  const requested: Array<{ sessionId: unknown; parent: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (url: unknown, init?: { body?: string }) => {
    if (!String(url).includes("/completion"))
      return originalFetch(url as Parameters<typeof fetch>[0]);
    const payload = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    requested.push({ sessionId: payload.chat_session_id, parent: payload.parent_message_id });
    return new Response(
      'data: {"p":"response/fragments","o":"APPEND","v":' +
      '[{"type":"RESPONSE","content":"salvaged answer"}]}\n\n' +
      'event: close\ndata: {}\n\n',
    );
  }) as typeof fetch;

  try {
    const body: RequestBody = {
      model: "deepseek-v4-flash",
      conversation: "00000000-0000-0000-0000-000000000001",
      parent_message_id: 48,
      messages: [{ role: "user", content: "fix it" }],
      tools: [{
        type: "function",
        function: { name: "run", parameters: { type: "object", properties: {} } },
      }],
    };
    const seen: string[] = [];
    const running = client.streamChat(body, (chunk) => {
      const delta = chunk.choices[0]?.delta as { content?: string } | undefined;
      if (typeof delta?.content === "string") seen.push(delta.content);
    });
    if (!recover) await expect(running).rejects.toThrow(/deepseek-auth-token-invalid/);
    else {
      await running;
      expect(seen.join("")).toBe("salvaged answer");
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
  return { requested, refreshes };
}

describe("auth token refresh at prepare", () => {
  it("refreshes login once and completes after an invalid token", async () => {
    const { requested, refreshes } = await runAuthRefreshCase(true);
    expect(refreshes).toBe(1);
    expect(requested).toEqual([
      { sessionId: "00000000-0000-0000-0000-000000000001", parent: 48 },
    ]);
  });

  it("raises the auth error when the token stays invalid after refresh", async () => {
    const { requested, refreshes } = await runAuthRefreshCase(false);
    expect(refreshes).toBe(1);
    expect(requested).toEqual([]);
  });
});
