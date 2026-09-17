/** Shared live-turn core: streams reasoning/output deltas and detects tool calls. */
import type { ToolReasoningMode } from "../config/env.js";
import type { MessageId } from "./sessionStore.js";
import { StreamSieve, type SieveEvent } from "./streamSieve.js";
import { parseToolCalls, type OpenAIToolCall, type ToolDefHint } from "./toolCalls.js";
import { resolveToolTurn, type ToolTurnOutcome } from "./toolOutcome.js";
import { iterDeepSeekUpdates } from "./updates.js";

export interface StreamTurnHandlers {
  onReasoning?: (delta: string) => void;
  onText?: (delta: string) => void;
  onToolCalls?: (calls: OpenAIToolCall[]) => void;
}

export interface StreamTurnInput {
  upstream: Response;
  idSeed: string;
  reasoningMode: ToolReasoningMode;
  emptyFallback?: string;
  toolHints?: readonly ToolDefHint[] | undefined;
  handlers?: StreamTurnHandlers;
}

export interface StreamTurnResult {
  responseMessageId: MessageId;
  requestMessageId: MessageId;
  responseText: string;
  reasoningText: string;
  outcome: ToolTurnOutcome;
  toolCalls: OpenAIToolCall[];
  framesEmitted: number;
  title: string | null;
  tokens: number;
}

/**
 * Consume one upstream SSE turn while forwarding safe deltas immediately.
 * Reasoning visibility follows ``reasoningMode``; output prose passes through
 * an incremental sieve that only holds back potential tool-call blocks.
 */
export async function streamToolTurn(input: StreamTurnInput): Promise<StreamTurnResult> {
  const handlers = input.handlers ?? {};
  const outputSieve = new StreamSieve(input.idSeed, input.toolHints);
  // Clean mode strips tool protocol that leaked into reasoning in real time.
  const reasoningSieve = input.reasoningMode === "clean" ? new StreamSieve(input.idSeed) : null;
  let responseText = "";
  let reasoningText = "";
  let responseMessageId: MessageId = null;
  let requestMessageId: MessageId = null;
  let title: string | null = null;
  let tokens = 0;
  let framesEmitted = 0;
  let committed = false;
  const committedCalls: OpenAIToolCall[] = [];

  const emitReasoning = (delta: string): void => {
    if (!delta) return;
    handlers.onReasoning?.(delta);
    framesEmitted += 1;
  };
  const emitText = (delta: string): void => {
    if (!delta) return;
    handlers.onText?.(delta);
    framesEmitted += 1;
  };
  const callKey = (call: OpenAIToolCall): string =>
    `${call.function.name}\0${call.function.arguments}`;
  const commitCalls = (calls: OpenAIToolCall[]): void => {
    const known = new Set(committedCalls.map(callKey));
    const fresh = calls.filter((call) => !known.has(callKey(call)));
    if (fresh.length === 0) return;
    handlers.onToolCalls?.(fresh);
    framesEmitted += 1;
    committedCalls.push(...fresh);
    committed = true;
  };
  const forwardOutputEvents = (events: SieveEvent[]): void => {
    for (const event of events) {
      // Prose after the first tool block is post-submit noise; later tool
      // blocks (multi-call turns) must still go through.
      if (event.type === "text") {
        if (!committed) emitText(event.delta);
      } else {
        commitCalls(event.calls);
      }
    }
  };

  for await (const update of iterDeepSeekUpdates(input.upstream)) {
    if (update.type === "ready") {
      responseMessageId = update.responseMessageId ?? responseMessageId;
      requestMessageId = update.requestMessageId ?? requestMessageId;
    } else if (update.type === "title") {
      title = update.title;
    } else if (update.type === "tokens") {
      tokens = update.value;
    } else if (update.type === "reasoning" && update.delta) {
      reasoningText += update.delta;
      if (committed) continue;
      if (input.reasoningMode === "raw") {
        emitReasoning(update.delta);
      } else if (reasoningSieve) {
        for (const event of reasoningSieve.feed(update.delta)) {
          if (event.type === "text") emitReasoning(event.delta);
          // Tool calls leaked into THINK are recovered once at turn end.
        }
      }
    } else if (update.type === "output" && update.delta) {
      responseText += update.delta;
      if (!committed) forwardOutputEvents(outputSieve.feed(update.delta));
    }
  }

  if (!committed) forwardOutputEvents(outputSieve.flush());

  const outcome = resolveToolTurn(
    responseText,
    reasoningText,
    input.idSeed,
    input.emptyFallback ?? "",
    { tools: input.toolHints },
  );

  // Calls that only became visible from the full text (leaked into THINK or
  // trapped behind an unclosed block) join the ones already streamed.
  commitCalls(outcome.parsed.toolCalls);

  if (!committed) {
    // Hidden mode never showed the reasoning channel: promote its cleaned
    // prose to visible output so a reasoning-only turn is not left empty.
    if (input.reasoningMode === "hidden" && !responseText.trim()) {
      const cleanedReasoning = parseToolCalls(reasoningText).content;
      if (cleanedReasoning) emitText(cleanedReasoning);
    }
    // Last-resort fallback, but only when the client has seen nothing at all.
    if (outcome.recoverableEmpty && framesEmitted === 0 && input.emptyFallback) {
      emitText(input.emptyFallback);
    }
  }

  return {
    responseMessageId,
    requestMessageId,
    responseText,
    reasoningText,
    outcome,
    toolCalls: committedCalls,
    framesEmitted,
    title,
    tokens,
  };
}
