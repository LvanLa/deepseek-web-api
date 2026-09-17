/** Routes the small OpenAI-compatible HTTP surface without a framework dependency. */
import type { IncomingMessage, ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";

import { handleChatCompletions } from "../api/chatCompletions.js";
import { handleModels } from "../api/models.js";
import { handleResponses } from "../api/responses.js";
import type { RequestBody } from "../deepseek/types.js";
import { errorMessage, errorStatus } from "../utils/errors.js";
import { readJsonBody, writeCorsPreflight, writeJson } from "../utils/http.js";
import { requireApiKey } from "./authMiddleware.js";
import type { ServerDependencies } from "./types.js";

function pathname(request: IncomingMessage): string {
  return new URL(request.url || "/", "http://localhost").pathname;
}

function logRequest(
  dependencies: ServerDependencies,
  request: IncomingMessage,
  path: string,
  body: RequestBody,
): void {
  const metadata = body.metadata && typeof body.metadata === "object" ? body.metadata : {};
  const items = Array.isArray(body.messages)
    ? body.messages
    : Array.isArray(body.input)
      ? body.input
      : [];
  const itemSummary = items.map((item) => {
    if (!item || typeof item !== "object") return { type: typeof item, chars: String(item ?? "").length };
    const value = item as Record<string, unknown>;
    return {
      role: typeof value.role === "string" ? value.role : undefined,
      type: typeof value.type === "string" ? value.type : undefined,
      chars: JSON.stringify(item)?.length ?? 0,
    };
  });
  dependencies.logger?.debug("API request received", {
    requestId: randomUUID(),
    method: request.method,
    path,
    model: typeof body.model === "string" ? body.model : undefined,
    userAgent: request.headers["user-agent"] ?? undefined,
    stream: body.stream === true,
    messages: Array.isArray(body.messages) ? body.messages.length : 0,
    inputItems: Array.isArray(body.input) ? body.input.length : 0,
    inputType: typeof body.input,
    tools: Array.isArray(body.tools) ? body.tools.length : 0,
    functions: Array.isArray(body.functions) ? body.functions.length : 0,
    itemSummary,
    hasSessionId: Boolean(
      body.chat_session_id ?? body.conversation ?? body.conversation_id ??
      (metadata as Record<string, unknown>).chat_session_id ??
      (metadata as Record<string, unknown>).conversation_id,
    ),
    hasPreviousResponseId: Boolean(
      body.previous_response_id ?? body.previous_response ??
      (metadata as Record<string, unknown>).previous_response_id,
    ),
  });
}

/** Apply CORS and authentication before dispatching exact method/path pairs. */
export async function routeRequest(
  request: IncomingMessage,
  response: ServerResponse,
  dependencies: ServerDependencies,
): Promise<void> {
  if (request.method === "OPTIONS") {
    writeCorsPreflight(response);
    return;
  }

  const path = pathname(request);
  if (request.method === "GET" && path === "/health") {
    writeJson(response, 200, { ok: true });
    return;
  }
  if (path.startsWith("/v1/") && !requireApiKey(request, response, dependencies.apiKeys)) return;

  if (request.method === "GET" && path === "/v1/models") {
    handleModels(response);
    return;
  }
  if (request.method === "POST" && path === "/v1/responses") {
    const body: RequestBody = await readJsonBody(request);
    logRequest(dependencies, request, path, body);
    await handleResponses(response, body, dependencies.client);
    return;
  }
  if (request.method === "POST" && path === "/v1/chat/completions") {
    const body: RequestBody = await readJsonBody(request);
    logRequest(dependencies, request, path, body);
    await handleChatCompletions(response, body, dependencies.client);
    return;
  }
  writeJson(response, 404, { error: { message: "not found" } });
}

/** Avoid writing a second JSON header after a streaming response has started. */
export function handleRouteError(
  response: ServerResponse,
  error: unknown,
  debug: boolean,
): void {
  if (response.headersSent) {
    response.end();
    return;
  }
  writeJson(response, errorStatus(error), {
    error: {
      message: errorMessage(error),
      ...(debug && error instanceof Error ? { stack: error.stack } : {}),
    },
  });
}
