/** Maps normalized DeepSeek updates to OpenAI Responses objects and SSE events. */
import type { ToolReasoningMode } from "../config/env.js";
import { ResponseEventWriter, type ResponseEmitter } from "./responseEvents.js";
import type { MessageId } from "./sessionStore.js";
import { streamToolTurn } from "./streamTurn.js";
import type { OpenAIToolCall, ToolDefHint } from "./toolCalls.js";
import { canonicalParsedAssistantText, parseToolCalls } from "./toolCalls.js";
import { EMPTY_TOOL_RESPONSE_TEXT, type ToolTurnOutcome } from "./toolOutcome.js";
import type { ModelType, PublicModel } from "./types.js";
import { iterDeepSeekUpdates } from "./updates.js";
export type { ResponseEmitter } from "./responseEvents.js";

export interface ResponseMetadata extends Record<string, unknown> {
  chat_session_id: string;
  source: "chat.deepseek.com";
  thinking_enabled?: boolean;
  search_enabled?: boolean;
  model_type?: ModelType;
}
export interface OpenAIResponse {
  id: string;
  object: "response";
  created_at: number;
  status: "in_progress" | "completed";
  model: PublicModel;
  output: Array<Record<string, unknown>>;
  output_text?: string;
  title?: string;
  usage: { input_tokens: number; output_tokens: number; total_tokens: number };
  metadata: ResponseMetadata;
}
export interface CompletionDiagnostics {
  reasoningChars: number; outputChars: number; toolCallCount: number;
  emptyUpstream: boolean; recoverableEmpty: boolean; promotedReasoning: boolean;
  upstreamTrace?: string[];
}
export interface MappedResponseResult {
  response: OpenAIResponse;
  requestMessageId: MessageId;
  responseMessageId: MessageId;
  rawOutputText: string;
  diagnostics: CompletionDiagnostics;
  framesEmitted: number;
}

type ResponseInput = {
  upstream: Response;
  publicModel: PublicModel;
  modelType: ModelType;
  sessionId: string;
  thinkingEnabled: boolean;
  searchEnabled: boolean;
  toolCompatibilityEnabled: boolean;
  toolReasoning: ToolReasoningMode;
  emptyToolResponseText?: string;
  toolHints?: readonly ToolDefHint[] | undefined;
  emit?: ResponseEmitter;
};

type ResponseIds = { requestMessageId: MessageId; responseMessageId: MessageId };

function responseBase(id: string, publicModel: PublicModel, sessionId: string, createdAt: number): OpenAIResponse {
  return {
    id, object: "response", created_at: createdAt, status: "in_progress", model: publicModel,
    output: [],
    usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
    metadata: { chat_session_id: sessionId, source: "chat.deepseek.com" },
  };
}

function functionCallItem(call: OpenAIToolCall): Record<string, unknown> {
  return {
    type: "function_call", id: `fc_${call.id.slice(5)}`, call_id: call.id,
    name: call.function.name, arguments: call.function.arguments, status: "completed",
  };
}

function buildMetadata(input: ResponseInput, meta: ResponseIds & { title: string | null }): ResponseMetadata {
  return {
    chat_session_id: input.sessionId,
    source: "chat.deepseek.com",
    title: meta.title,
    thinking_enabled: input.thinkingEnabled,
    search_enabled: input.searchEnabled,
    model_type: input.modelType,
    request_message_id: meta.requestMessageId,
    response_message_id: meta.responseMessageId,
    thinking_levels_supported: false,
    tool_compatibility: input.toolCompatibilityEnabled,
    ...(input.modelType === "expert" ? { search_note: "expert mode does not support search on web" } : {}),
  };
}

function toDiagnostics(
  reasoning: string,
  outputText: string,
  outcome: ToolTurnOutcome,
  toolCallCount: number,
  upstreamTrace?: string[],
): CompletionDiagnostics {
  return {
    reasoningChars: reasoning.length,
    outputChars: outputText.length,
    toolCallCount,
    emptyUpstream: outcome.emptyUpstream,
    recoverableEmpty: outcome.recoverableEmpty, promotedReasoning: outcome.promotedReasoning,
    ...(upstreamTrace ? { upstreamTrace } : {}),
  };
}

/** Consume one upstream stream while building both live events and the final response. */
export async function consumeResponses(input: ResponseInput): Promise<MappedResponseResult> {
  const id = `resp_${input.sessionId}`;
  const writer = new ResponseEventWriter(input.emit, `${id}_reasoning`, `${id}_message`);
  const createdAt = Math.floor(Date.now() / 1000);
  writer.start(responseBase(id, input.publicModel, input.sessionId, createdAt));

  if (!input.toolCompatibilityEnabled) {
    return await consumePlainResponses(input, id, createdAt, writer);
  }
  return await consumeToolResponses(input, id, createdAt, writer);
}

/** Tool-compatible turn: live reasoning/text deltas, function calls on capture. */
async function consumeToolResponses(
  input: ResponseInput,
  id: string,
  createdAt: number,
  writer: ResponseEventWriter,
): Promise<MappedResponseResult> {
  const live = Boolean(input.emit);
  const output: Array<Record<string, unknown>> = [];
  const emittedCallIds = new Set<string>();
  let streamedReasoning = "";
  let streamedOutput = "";

  const emitFunctionCall = (call: OpenAIToolCall): void => {
    const item = functionCallItem(call);
    const outputIndex = output.length;
    output.push(item);
    writer.emitFunctionCall(item, outputIndex);
    emittedCallIds.add(call.id);
  };

  const turn = await streamToolTurn({
    upstream: input.upstream,
    idSeed: id,
    reasoningMode: input.toolReasoning,
    emptyFallback: input.emptyToolResponseText ?? EMPTY_TOOL_RESPONSE_TEXT,
    toolHints: input.toolHints,
    ...(live
      ? {
        handlers: {
          onReasoning: (delta: string) => {
            streamedReasoning += delta;
            writer.emitReasoningDelta(delta);
          },
          onText: (delta: string) => {
            streamedOutput += delta;
            writer.emitOutputDelta(delta);
          },
          onToolCalls: (calls: OpenAIToolCall[]) => {
            writer.finishOpenItems(streamedReasoning, streamedOutput, output);
            for (const call of calls) emitFunctionCall(call);
          },
        },
      }
      : {}),
  });

  const parsed = turn.outcome.parsed;
  const hasToolCalls = parsed.toolCalls.length > 0;
  const hasResponseText = turn.responseText.trim().length > 0;

  let visibleText: string;
  if (live) {
    // Deltas already streamed: close any items still open, then append calls
    // recovered after the stream ended (e.g. protocol leaked into THINK).
    if (streamedReasoning && input.toolReasoning !== "hidden") {
      writer.finishReasoning(streamedReasoning, output);
    }
    if (writer.messageOpened) writer.finishMessage(streamedOutput, output);
    for (const call of turn.toolCalls) {
      if (!emittedCallIds.has(call.id)) emitFunctionCall(call);
    }
    visibleText = streamedOutput;
  } else {
    // Buffered non-streaming assembly: preserve promotion/visibility rules.
    visibleText = parsed.content;
    let reasoningVisible = "";
    if (hasToolCalls) {
      if (input.toolReasoning === "clean") reasoningVisible = parseToolCalls(turn.reasoningText).content;
    } else if (!turn.outcome.promotedReasoning && hasResponseText) {
      reasoningVisible = turn.reasoningText;
    }
    const includeMessage = !hasToolCalls || visibleText.length > 0;
    if (reasoningVisible && !writer.reasoningOpened) writer.emitReasoningDelta(reasoningVisible);
    if (includeMessage && !writer.messageOpened) writer.emitOutputDelta(visibleText || "");
    writer.finishReasoning(reasoningVisible, output);
    writer.finishMessage(visibleText, output);
    for (const call of parsed.toolCalls) emitFunctionCall(call);
  }

  // Canonical storage text: canonical tool markup for tool turns, otherwise the
  // resolved (promoted/fallback) content, never raw upstream protocol garbage.
  const rawOutputText = hasToolCalls ? canonicalParsedAssistantText(parsed) : parsed.content;
  const final: OpenAIResponse = {
    id, object: "response", created_at: createdAt, status: "completed",
    model: input.publicModel,
    output,
    output_text: visibleText,
    ...(turn.title ? { title: turn.title } : {}),
    usage: { input_tokens: 0, output_tokens: turn.tokens, total_tokens: turn.tokens },
    metadata: buildMetadata(input, {
      requestMessageId: turn.requestMessageId, responseMessageId: turn.responseMessageId, title: turn.title,
    }),
  };
  writer.emit("response.completed", { type: "response.completed", response: final });
  return {
    response: final,
    requestMessageId: turn.requestMessageId,
    responseMessageId: turn.responseMessageId,
    rawOutputText,
    diagnostics: toDiagnostics(
      turn.reasoningText,
      turn.responseText,
      turn.outcome,
      parsed.toolCalls.length,
      turn.upstreamTrace,
    ),
    framesEmitted: live ? turn.framesEmitted : 0,
  };
}

/** Plain (tool-free) turn: both channels are forwarded live as they arrive. */
async function consumePlainResponses(
  input: ResponseInput, id: string, createdAt: number, writer: ResponseEventWriter,
): Promise<MappedResponseResult> {
  const live = Boolean(input.emit);
  let reasoning = "";
  let outputText = "";
  let title: string | null = null;
  let tokens = 0;
  const ids: ResponseIds = { requestMessageId: null, responseMessageId: null };

  for await (const update of iterDeepSeekUpdates(input.upstream)) {
    if (update.type === "ready") {
      ids.requestMessageId = update.requestMessageId ?? ids.requestMessageId;
      ids.responseMessageId = update.responseMessageId ?? ids.responseMessageId;
    } else if (update.type === "title") {
      title = update.title;
    } else if (update.type === "tokens") {
      tokens = update.value;
    } else if (update.type === "reasoning" && update.delta) {
      reasoning += update.delta;
      writer.emitReasoningDelta(update.delta);
    } else if (update.type === "output" && update.delta) {
      outputText += update.delta;
      writer.emitOutputDelta(update.delta);
    }
  }

  const hasResponseText = outputText.trim().length > 0;
  // Streaming keeps a reasoning-only turn on the reasoning channel; buffered
  // mode promotes it to the message so the non-streaming JSON stays complete.
  const visibleText = hasResponseText ? outputText : live ? "" : reasoning;
  const reasoningVisible = reasoning ? (hasResponseText || live ? reasoning : "") : "";

  if (!live) {
    if (reasoningVisible && !writer.reasoningOpened) writer.emitReasoningDelta(reasoningVisible);
    if (visibleText && !writer.messageOpened) writer.emitOutputDelta(visibleText);
    if (!writer.messageOpened) writer.emitOutputDelta("");
  }

  const output: Array<Record<string, unknown>> = [];
  if (reasoningVisible) writer.finishReasoning(reasoningVisible, output);
  writer.finishMessage(visibleText, output);

  const final: OpenAIResponse = {
    id, object: "response", created_at: createdAt, status: "completed",
    model: input.publicModel, output, output_text: visibleText,
    ...(title ? { title } : {}),
    usage: { input_tokens: 0, output_tokens: tokens, total_tokens: tokens },
    metadata: buildMetadata(input, { ...ids, title }),
  };
  writer.emit("response.completed", { type: "response.completed", response: final });
  return {
    response: final, ...ids,
    rawOutputText: hasResponseText ? outputText : reasoning,
    diagnostics: {
      reasoningChars: reasoning.length, outputChars: outputText.length, toolCallCount: 0,
      emptyUpstream: !reasoning.trim() && !outputText.trim(),
      recoverableEmpty: false,
      promotedReasoning: !hasResponseText && Boolean(reasoning.trim()),
    },
    framesEmitted: 0,
  };
}
