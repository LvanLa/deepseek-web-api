/** Orchestrates login state, session reuse, PoW, upstream streaming, and persistence. */
import type { LoginManager } from "../browser/login.js";
import type { AppConfig } from "../config/env.js";
import { isRecord } from "../utils/json.js";
import type { Logger } from "../utils/logger.js";
import {
  finalChunk,
  streamPlainChat,
  streamToolChat,
  toolDiagnostics,
  toolSessionText,
  type ChatStreamChunk,
} from "./chatStream.js";
import { prepareRun, type PreparedRun } from "./prepareRun.js";
import {
  catchContextLength,
  frameTracker,
  isAuthTokenError,
  logContextFork,
  type FrameTracker,
} from "./recovery.js";
import {
  consumeResponses,
  type CompletionDiagnostics,
  type MappedResponseResult,
  type ResponseEmitter,
} from "./mapResponses.js";
import type { MessageId, SessionStore } from "./sessionStore.js";
import { EMPTY_TOOL_RESPONSE_TEXT } from "./toolOutcome.js";
import type { RequestBody } from "./types.js";

interface CompletionAttempt<T> { value: T; diagnostics: CompletionDiagnostics; framesEmitted?: number }
interface CompletionResult<T> extends CompletionAttempt<T> { run: PreparedRun; retry: number }
export type { ChatStreamChunk } from "./chatStream.js";

export class DeepSeekClient {
  constructor(
    private readonly config: AppConfig,
    private readonly login: LoginManager,
    private readonly sessions: SessionStore,
    private readonly logger: Logger,
  ) {}

  async initialize(): Promise<void> {
    await this.login.ensureLoggedIn();
  }
  async completeResponses(body: RequestBody, emit?: ResponseEmitter): Promise<MappedResponseResult> {
    const initialRun = await this.prepareWithAuthRefresh(body);
    if (!initialRun.hasTools) {
      return this.withPlainRecovery(body, initialRun, (run, tracker) => {
        const tracked = emit ? this.trackedResponseEmit(emit, tracker) : undefined;
        return consumeResponses({
          ...this.responseInput(run),
          ...(tracked ? { emit: tracked } : {}),
        });
      }, (run, result) => {
        this.logAttempt(run, result.diagnostics);
        this.remember(run, result.responseMessageId, result.rawOutputText);
        this.setResponseMetadata(result, run, run.retry);
      });
    }

    const completed = await this.withToolRecovery(body, initialRun, async (run, _retry, final, tracker) => {
      const tracked = emit ? this.trackedResponseEmit(emit, tracker) : undefined;
      const result = await consumeResponses({
        ...this.responseInput(run),
        emptyToolResponseText: final ? EMPTY_TOOL_RESPONSE_TEXT : "",
        ...(tracked ? { emit: tracked } : {}),
      });
      return {
        value: { result },
        diagnostics: result.diagnostics,
        framesEmitted: result.framesEmitted,
      };
    });

    if (!completed.diagnostics.recoverableEmpty) {
      this.remember(
        completed.run,
        completed.value.result.responseMessageId,
        completed.value.result.rawOutputText,
      );
    }
    this.setResponseMetadata(completed.value.result, completed.run, completed.retry);
    return completed.value.result;
  }
  async streamChat(body: RequestBody, emit: (chunk: ChatStreamChunk) => void): Promise<void> {
    const initialRun = await this.prepareWithAuthRefresh(body);
    if (!initialRun.hasTools) {
      await this.withPlainRecovery(body, initialRun, (run, tracker) => {
        const tracked = (chunk: ChatStreamChunk) => { tracker.noteFrame(); emit(chunk); };
        return streamPlainChat(run, tracked);
      }, (run, result) => {
        this.logAttempt(run, result.diagnostics);
        this.remember(run, result.responseMessageId, result.finalText);
      });
      return;
    }
    const completed = await this.withToolRecovery(body, initialRun, async (run, _retry, final, tracker) => {
      const tracked = (chunk: ChatStreamChunk) => { tracker.noteFrame(); emit(chunk); };
      const value = await streamToolChat(
        run,
        this.config.toolReasoning,
        final ? EMPTY_TOOL_RESPONSE_TEXT : "",
        tracked,
      );
      return {
        value,
        diagnostics: toolDiagnostics(value.outcome, value.upstreamTrace),
        framesEmitted: value.framesEmitted,
      };
    });
    emit(finalChunk(completed.run, Math.floor(Date.now() / 1000), completed.value.finishReason));
    if (!completed.diagnostics.recoverableEmpty) {
      this.remember(
        completed.run,
        completed.value.responseMessageId,
        toolSessionText(completed.value.outcome),
      );
    }
  }
  private async withToolRecovery<T>(
    body: RequestBody,
    initialRun: PreparedRun,
    execute: (run: PreparedRun, retry: number, finalAttempt: boolean, tracker: FrameTracker)
      => Promise<CompletionAttempt<T>>,
  ): Promise<CompletionResult<T>> {
    const tracker = frameTracker();
    let first: CompletionAttempt<T>;
    let contextError = false;
    try {
      first = await execute(initialRun, 0, false, tracker);
    } catch (error) {
      if (catchContextLength(error, tracker.frames) === null) throw error;
      contextError = true;
    }
    // Once deltas streamed to the client they cannot be taken back, so a live
    // turn with any emitted frame is never silently retried.
    if (!contextError) {
      this.logAttempt(initialRun, first!.diagnostics);
      if (!first!.diagnostics.recoverableEmpty || (first!.framesEmitted ?? 0) > 0) {
        return { ...first!, run: initialRun, retry: 0 };
      }
      this.logger.info(
        "Retrying unusable DeepSeek tool response", this.logFields(initialRun, first!.diagnostics),
      );
    }
    // Empty/context-length failures on a reused session mean its accumulated
    // history overflowed; fork a new session, else compact in place.
    const freshSession = initialRun.reusedSession;
    if (contextError) {
      logContextFork(this.logger, this.forkLogFields(initialRun), freshSession);
    } else if (first!.diagnostics.emptyUpstream) {
      this.logger.info(
        "Retrying in a fresh DeepSeek session after silent empty upstream",
        this.logFields(initialRun, first!.diagnostics),
      );
    }
    const parent = contextError ? null : this.retryParentId(first!.value);
    const retryRun = await this.prepareWithAuthRefresh(body, initialRun, parent, freshSession);
    const second = await execute(retryRun, 1, true, tracker);
    this.logAttempt(retryRun, second.diagnostics);
    if (second.diagnostics.recoverableEmpty) {
      this.logger.warn("DeepSeek tool response remained unusable after retry", {
        ...this.logFields(retryRun, second.diagnostics),
        fallbackVisible: true,
      });
    } else {
      this.logger.info("DeepSeek tool response recovered", this.logFields(retryRun, second.diagnostics));
    }
    return { ...second, run: retryRun, retry: 1 };
  }

  /** One fork/compact retry for a non-tool turn that hit the length limit. */
  private async withPlainRecovery<T>(
    body: RequestBody,
    initialRun: PreparedRun,
    consume: (run: PreparedRun, tracker: FrameTracker) => Promise<T>,
    settle: (run: PreparedRun, value: T) => void,
  ): Promise<T> {
    const tracker = frameTracker();
    try {
      const value = await consume(initialRun, tracker);
      settle(initialRun, value);
      return value;
    } catch (error) {
      if (catchContextLength(error, tracker.frames) === null) throw error;
      const freshSession = initialRun.reusedSession;
      logContextFork(this.logger, this.forkLogFields(initialRun), freshSession);
      const retryRun = await this.prepareWithAuthRefresh(body, initialRun, null, freshSession);
      const value = await consume(retryRun, tracker);
      settle(retryRun, value);
      return value;
    }
  }

  /** Wrap a Responses emitter so every delivered frame is counted. */
  private trackedResponseEmit(emit: ResponseEmitter, tracker: FrameTracker): ResponseEmitter {
    return (event: string, data: Record<string, unknown>) => { tracker.noteFrame(); emit(event, data); };
  }

  /** Compact log fields for a fork decided without a diagnostics result. */
  private forkLogFields(run: PreparedRun): Record<string, unknown> {
    return { sessionId: run.sessionId, reused: run.reusedSession, promptChars: run.prompt.length, retry: run.retry };
  }

  /** Prepare once; on a rejected token, refresh login and prepare a second time. */
  private async prepareWithAuthRefresh(
    body: RequestBody, retryFrom?: PreparedRun, retryParent: MessageId = null, freshSession = false,
  ): Promise<PreparedRun> {
    try {
      return await this.prepare(body, retryFrom, retryParent, freshSession);
    } catch (error) {
      if (!isAuthTokenError(error)) throw error;
      this.logger.info("DeepSeek token rejected; refreshing login and retrying", {
        error: error instanceof Error ? error.message : String(error),
      });
      await this.login.refreshLogin();
      return this.prepare(body, retryFrom, retryParent, freshSession);
    }
  }

  /** Thin wrapper over prepareRun with this client's dependencies. */
  private prepare(
    body: RequestBody, retryFrom?: PreparedRun, retryParent: MessageId = null, freshSession = false,
  ): Promise<PreparedRun> {
    return prepareRun(
      { config: this.config, login: this.login, sessions: this.sessions, logger: this.logger },
      body, retryFrom, retryParent, freshSession,
    );
  }
  private responseInput(run: PreparedRun) {
    return {
      upstream: run.upstream, publicModel: run.publicModel, modelType: run.modelType,
      sessionId: run.sessionId, thinkingEnabled: run.thinking, searchEnabled: run.search,
      toolCompatibilityEnabled: run.hasTools, toolReasoning: this.config.toolReasoning,
      toolHints: run.toolHints,
    };
  }
  private setResponseMetadata(result: MappedResponseResult, run: PreparedRun, retry: number): void {
    result.response.metadata.reused_session = run.reusedSession;
    result.response.metadata.parent_message_id = run.parentMessageId;
    result.response.metadata.empty_response_retry = retry;
  }

  private logAttempt(run: PreparedRun, diagnostics: CompletionDiagnostics): void {
    const fields = this.logFields(run, diagnostics);
    this.logger.debug("DeepSeek completion result", fields);
    if (diagnostics.emptyUpstream) this.logger.info("DeepSeek upstream returned empty channels", fields);
  }

  private logFields(run: PreparedRun, diagnostics: CompletionDiagnostics): Record<string, unknown> {
    return {
      sessionId: run.sessionId, reused: run.reusedSession, parentMessageId: run.parentMessageId,
      promptChars: run.prompt.length, retry: run.retry,
      reasoningChars: diagnostics.reasoningChars, outputChars: diagnostics.outputChars,
      toolCallCount: diagnostics.toolCallCount, emptyUpstream: diagnostics.emptyUpstream,
      ...(diagnostics.upstreamTrace ? { upstreamTrace: diagnostics.upstreamTrace } : {}),
    };
  }

  /** Extract the upstream response id used to chain a silent empty-turn retry. */
  private retryParentId(value: unknown): MessageId {
    const source = isRecord(value) && isRecord(value.result) ? value.result : value;
    return isRecord(source) &&
      (typeof source.responseMessageId === "string" || typeof source.responseMessageId === "number")
      ? source.responseMessageId
      : null;
  }

  private remember(run: PreparedRun, responseMessageId: MessageId, responseText: string): void {
    // A trailing assistant turn is replayed history, not a new request turn;
    // exclude it so the new response is not stored as a second assistant.
    const fullTurns = run.allTurns.at(-1)?.role === "assistant"
      ? run.allTurns.slice(0, -1)
      : run.allTurns;
    this.sessions.remember({
      sessionId: run.sessionId, modelType: run.modelType, responseMessageId,
      convKey: run.convKey, fullTurns, assistantContent: responseText,
      instructionFingerprint: run.instructionFingerprint, toolsFingerprint: run.toolsFingerprint,
    });
  }
}
