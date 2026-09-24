/** Converts DeepSeek patch variants into stable reasoning/output update events. */
import { HttpError } from "../utils/errors.js";
import { isRecord } from "../utils/json.js";
import type { DeepSeekFragment, DeepSeekSseEvent } from "./types.js";
import { iterDeepSeekSse } from "./sse.js";

export type DeepSeekUpdate =
  | { type: "ready"; requestMessageId: string | number | null; responseMessageId: string | number | null }
  | { type: "title"; title: string }
  | { type: "reasoning"; delta: string }
  | { type: "output"; delta: string }
  | { type: "tokens"; value: number }
  | { type: "close" };

function fragments(value: unknown): DeepSeekFragment[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function messageId(value: unknown): string | number | null {
  return typeof value === "string" || typeof value === "number" ? value : null;
}

function fragmentUpdate(fragment: DeepSeekFragment): DeepSeekUpdate | null {
  const type = typeof fragment.type === "string" ? fragment.type : "";
  const content = typeof fragment.content === "string" ? fragment.content : "";
  if (type === "THINK") return { type: "reasoning", delta: content };
  if (type === "RESPONSE") return { type: "output", delta: content };
  if (type === "SEARCH" || type === "SEARCH_REF") {
    return { type: "reasoning", delta: content ? `\n[search] ${content}\n` : "\n[search]\n" };
  }
  return null;
}

function nestedResponse(data: Record<string, unknown>): Record<string, unknown> | null {
  if (!isRecord(data.v) || !isRecord(data.v.response)) return null;
  return data.v.response;
}

function readyUpdate(event: DeepSeekSseEvent): DeepSeekUpdate {
  return {
    type: "ready",
    requestMessageId: messageId(event.data.request_message_id),
    responseMessageId: messageId(event.data.response_message_id),
  };
}

export type RawFrameSink = (entry: { event: string | null; raw: string }) => void;

/**
 * Track the active fragment type so content-only APPEND patches stay in the correct channel.
 * This is what prevents reasoning text from leaking into final output text.
 */
export async function* iterDeepSeekUpdates(
  response: Response,
  onRawFrame?: RawFrameSink,
): AsyncGenerator<DeepSeekUpdate> {
  let current: "reasoning" | "output" = "reasoning";
  for await (const event of iterDeepSeekSse(response)) {
    onRawFrame?.({ event: event.event, raw: event.raw ?? JSON.stringify(event.data) ?? "" });
    const { data } = event;
    if (event.event === "ready") {
      yield readyUpdate(event);
      continue;
    }
    if (event.event === "title") {
      if (typeof data.content === "string") yield { type: "title", title: data.content };
      continue;
    }
    // Upstream failures still arrive with HTTP 200; surface the real message
    // instead of ending empty, which would burn the automatic recovery retry.
    if (isContextLengthFrame(data)) {
      // Marked so the client can fork to a fresh session and retry once.
      throw new HttpError(
        502, `DeepSeek upstream error: ${upstreamErrorMessage(data)}`, "context_length_exceeded",
      );
    }
    if (isUpstreamError(event, data)) {
      throw new HttpError(502, `DeepSeek upstream error: ${upstreamErrorMessage(data)}`);
    }
    if (event.event === "close") {
      yield { type: "close" };
      break;
    }

    const responseData = nestedResponse(data);
    if (responseData) {
      for (const fragment of fragments(responseData.fragments)) {
        const update = fragmentUpdate(fragment);
        if (!update) continue;
        if (update.type === "reasoning" || update.type === "output") current = update.type;
        yield update;
      }
      if (typeof responseData.accumulated_token_usage === "number") {
        yield { type: "tokens", value: responseData.accumulated_token_usage };
      }
      continue;
    }

    if (data.p === "response/fragments" && data.o === "APPEND") {
      for (const fragment of fragments(data.v)) {
        const update = fragmentUpdate(fragment);
        if (!update) continue;
        if (update.type === "reasoning" || update.type === "output") current = update.type;
        yield update;
      }
      continue;
    }

    const isContentAppend = data.p === "response/fragments/-1/content" && data.o === "APPEND";
    const isRootAppend = data.p === undefined && data.o === "APPEND" && data.v !== undefined;
    const isBareString = data.p === undefined && data.o === undefined && typeof data.v === "string";
    if (isContentAppend || isRootAppend || isBareString) {
      const delta = String(data.v ?? "");
      if (delta) yield { type: current, delta };
      continue;
    }

    if (data.p === "response" && data.o === "BATCH" && Array.isArray(data.v)) {
      for (const patch of data.v) {
        if (isRecord(patch) && patch.p === "accumulated_token_usage" && typeof patch.v === "number") {
          yield { type: "tokens", value: patch.v };
        }
      }
    }
  }
}

/** True for an explicit or text-signaled context-length failure frame. */
function isContextLengthFrame(data: Record<string, unknown>): boolean {
  if (data.finish_reason === "context_length_exceeded") return true;
  const text = [data.content, data.message, data.msg].filter((v) => typeof v === "string").join(" ");
  return /context[ _-]?length|context_length|对话长度|长度上限|开启新对话/i.test(text);
}

/** Recognize the error shapes DeepSeek sends inside an otherwise 200 SSE stream. */
function isUpstreamError(event: DeepSeekSseEvent, data: Record<string, unknown>): boolean {
  if (event.event === "error" || data.type === "error" || data.o === "ERROR") return true;
  if (isRecord(data.error)) return true;
  return typeof data.code === "number" && data.code !== 0 &&
    (typeof data.message === "string" || typeof data.msg === "string");
}

/** Pull the human-readable message out of the known error payload variants. */
function upstreamErrorMessage(data: Record<string, unknown>): string {
  const candidates: unknown[] = [data.message, data.msg, data.content, data.error];
  if (isRecord(data.v)) candidates.push(data.v.message, data.v.msg, data.v.error, data.v.errmsg);
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate;
    if (isRecord(candidate) && typeof candidate.message === "string" && candidate.message) {
      return candidate.message;
    }
  }
  return JSON.stringify(data).slice(0, 300);
}

/** Whether a caught failure is a marked context-length upstream error. */
export function isContextLengthExceeded(error: unknown): boolean {
  return error instanceof HttpError && error.code === "context_length_exceeded";
}
