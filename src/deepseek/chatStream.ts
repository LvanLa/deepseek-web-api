/** Maps DeepSeek updates to OpenAI-compatible Chat Completions chunks. */
import type { ToolReasoningMode } from "../config/env.js";
import type { CompletionDiagnostics } from "./mapResponses.js";
import type { MessageId } from "./sessionStore.js";
import { streamToolTurn } from "./streamTurn.js";
import { canonicalParsedAssistantText, type ToolDefHint } from "./toolCalls.js";
import { type ToolTurnOutcome } from "./toolOutcome.js";
import type { PublicModel } from "./types.js";
import { iterDeepSeekUpdates } from "./updates.js";

export interface ChatRun {
  upstream: Response;
  sessionId: string;
  publicModel: PublicModel;
  toolHints?: readonly ToolDefHint[];
}

export interface ChatStreamChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: PublicModel;
  choices: Array<{
    index: number;
    delta: Record<string, unknown>;
    finish_reason: "stop" | "tool_calls" | null;
  }>;
  conversation?: string;
  previous_response_id?: string;
}

export interface PlainChatResult {
  responseMessageId: MessageId;
  finalText: string;
  diagnostics: CompletionDiagnostics;
}

export interface ToolStreamValue {
  outcome: ToolTurnOutcome;
  responseMessageId: MessageId;
  framesEmitted: number;
  finishReason: "stop" | "tool_calls";
  upstreamTrace: string[];
}

function chatChunk(
  run: ChatRun,
  created: number,
  delta: Record<string, unknown>,
  finishReason: "stop" | "tool_calls" | null,
): ChatStreamChunk {
  return {
    id: `chatcmpl_${run.sessionId}`,
    object: "chat.completion.chunk",
    created,
    model: run.publicModel,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

export function finalChunk(
  run: ChatRun,
  created: number,
  finishReason: "stop" | "tool_calls",
): ChatStreamChunk {
  return {
    ...chatChunk(run, created, {}, finishReason),
    conversation: run.sessionId,
    previous_response_id: `resp_${run.sessionId}`,
  };
}

/** Plain (tool-free) turn: reasoning and answer deltas are both forwarded live. */
export async function streamPlainChat(
  run: ChatRun,
  emit: (chunk: ChatStreamChunk) => void,
): Promise<PlainChatResult> {
  const created = Math.floor(Date.now() / 1000);
  let responseText = "";
  let reasoningText = "";
  let responseMessageId: MessageId = null;
  let roleSent = false;
  const writeDelta = (delta: Record<string, unknown>): void => {
    const withRole = roleSent ? delta : { role: "assistant", ...delta };
    roleSent = true;
    emit(chatChunk(run, created, withRole, null));
  };

  for await (const update of iterDeepSeekUpdates(run.upstream)) {
    if (update.type === "ready") {
      responseMessageId = update.responseMessageId ?? responseMessageId;
    } else if (update.type === "reasoning" && update.delta) {
      reasoningText += update.delta;
      writeDelta({ reasoning_content: update.delta });
    } else if (update.type === "output" && update.delta) {
      responseText += update.delta;
      writeDelta({ content: update.delta });
    }
  }

  // A reasoning-only turn keeps the reasoning channel; never duplicate it as content.
  const finalText = responseText.trim() ? responseText : reasoningText;
  emit(finalChunk(run, created, "stop"));
  return {
    responseMessageId,
    finalText,
    diagnostics: {
      reasoningChars: reasoningText.length,
      outputChars: responseText.length,
      toolCallCount: 0,
      emptyUpstream: !reasoningText.trim() && !responseText.trim(),
      recoverableEmpty: false,
      promotedReasoning: !responseText.trim() && Boolean(reasoningText.trim()),
    },
  };
}

/**
 * Tool-compatible turn streamed live: reasoning deltas follow ``toolReasoning``,
 * safe prose is forwarded immediately, and a complete tool block is emitted as
 * standard tool_calls deltas. The terminal finish chunk is left to the caller
 * so an internal zero-frame retry does not leak a premature stop.
 */
export async function streamToolChat(
  run: ChatRun,
  toolReasoning: ToolReasoningMode,
  emptyFallback: string,
  emit: (chunk: ChatStreamChunk) => void,
): Promise<ToolStreamValue> {
  const created = Math.floor(Date.now() / 1000);
  let roleSent = false;
  let toolIndex = 0;
  const writeDelta = (delta: Record<string, unknown>): void => {
    const withRole = roleSent ? delta : { role: "assistant", ...delta };
    roleSent = true;
    emit(chatChunk(run, created, withRole, null));
  };

  const result = await streamToolTurn({
    upstream: run.upstream,
    idSeed: `chatcmpl_${run.sessionId}`,
    reasoningMode: toolReasoning,
    emptyFallback,
    toolHints: run.toolHints,
    handlers: {
      onReasoning: (delta) => writeDelta({ reasoning_content: delta }),
      onText: (delta) => writeDelta({ content: delta }),
      onToolCalls: (calls) => {
        writeDelta({ tool_calls: calls.map((call) => ({ index: toolIndex++, ...call })) });
      },
    },
  });

  return {
    outcome: result.outcome,
    responseMessageId: result.responseMessageId,
    framesEmitted: result.framesEmitted,
    finishReason: result.toolCalls.length > 0 ? "tool_calls" : "stop",
    upstreamTrace: result.upstreamTrace,
  };
}

export function toolDiagnostics(outcome: ToolTurnOutcome, upstreamTrace?: string[]): CompletionDiagnostics {
  return {
    reasoningChars: outcome.reasoningText.length,
    outputChars: outcome.outputText.length,
    toolCallCount: outcome.parsed.toolCalls.length,
    emptyUpstream: outcome.emptyUpstream,
    recoverableEmpty: outcome.recoverableEmpty,
    promotedReasoning: outcome.promotedReasoning,
    ...(upstreamTrace ? { upstreamTrace } : {}),
  };
}

/** Canonical assistant text stored for session history after a tool turn. */
export function toolSessionText(outcome: ToolTurnOutcome): string {
  return outcome.parsed.toolCalls.length > 0
    ? canonicalParsedAssistantText(outcome.parsed)
    : outcome.parsed.content;
}
