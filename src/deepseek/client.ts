/** Orchestrates login state, session reuse, PoW, upstream streaming, and persistence. */
import type { LoginManager } from "../browser/login.js";
import type { AppConfig } from "../config/env.js";
import { HttpError } from "../utils/errors.js";
import { isRecord } from "../utils/json.js";
import type { Logger } from "../utils/logger.js";
import {
  finalChunk,
  streamPlainChat,
  streamToolChat,
  toolDiagnostics,
  toolSessionText,
  type ChatRun,
  type ChatStreamChunk,
} from "./chatStream.js";
import { openCompletionStream } from "./completion.js";
import {
  consumeResponses,
  type CompletionDiagnostics,
  type MappedResponseResult,
  type ResponseEmitter,
} from "./mapResponses.js";
import { resolveModel, resolveSearch, resolveThinking } from "./models.js";
import { prepareCompletion } from "./pow.js";
import { buildDeepSeekPrompt, buildToolRecoveryPrompt } from "./promptBuild.js";
import type { MessageId, SessionStore } from "./sessionStore.js";
import { EMPTY_TOOL_RESPONSE_TEXT } from "./toolOutcome.js";
import type { MessageTurn, ModelType, RequestBody } from "./types.js";

interface PreparedRun extends ChatRun {
  prompt: string; parentMessageId: MessageId; modelType: ModelType;
  thinking: boolean; search: boolean; reusedSession: boolean;
  convKey: string | null; requestTurns: MessageTurn[]; allTurns: MessageTurn[];
  instructionFingerprint: string; toolsFingerprint: string;
  latestUserText: string; hasTools: boolean; retry: number;
}

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
    const initialRun = await this.prepare(body);
    if (!initialRun.hasTools) {
      const result = await consumeResponses({
        ...this.responseInput(initialRun),
        ...(emit ? { emit } : {}),
      });
      this.logAttempt(initialRun, result.diagnostics);
      this.remember(initialRun, result.responseMessageId, result.rawOutputText);
      this.setResponseMetadata(result, initialRun, 0);
      return result;
    }

    const completed = await this.withToolRecovery(body, initialRun, async (run, _retry, final) => {
      const result = await consumeResponses({
        ...this.responseInput(run),
        emptyToolResponseText: final ? EMPTY_TOOL_RESPONSE_TEXT : "",
        ...(emit ? { emit } : {}),
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
    const initialRun = await this.prepare(body);
    if (!initialRun.hasTools) {
      const result = await streamPlainChat(initialRun, emit);
      this.logAttempt(initialRun, result.diagnostics);
      this.remember(initialRun, result.responseMessageId, result.finalText);
      return;
    }

    const completed = await this.withToolRecovery(body, initialRun, async (run, _retry, final) => {
      const value = await streamToolChat(
        run,
        this.config.toolReasoning,
        final ? EMPTY_TOOL_RESPONSE_TEXT : "",
        emit,
      );
      return { value, diagnostics: toolDiagnostics(value.outcome), framesEmitted: value.framesEmitted };
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
    execute: (
      run: PreparedRun,
      retry: number,
      finalAttempt: boolean,
    ) => Promise<CompletionAttempt<T>>,
  ): Promise<CompletionResult<T>> {
    const first = await execute(initialRun, 0, false);
    this.logAttempt(initialRun, first.diagnostics);
    // Once deltas streamed to the client they cannot be taken back, so a live
    // turn with any emitted frame is never silently retried.
    if (!first.diagnostics.recoverableEmpty || (first.framesEmitted ?? 0) > 0) {
      return { ...first, run: initialRun, retry: 0 };
    }

    this.logger.info("Retrying unusable DeepSeek tool response", this.logFields(initialRun, first.diagnostics));
    const retryRun = await this.prepare(body, initialRun, this.retryParentId(first.value));
    const second = await execute(retryRun, 1, true);
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

  /** Resolve trusted request options and preserve the last good parent across an empty retry. */
  private async prepare(
    body: RequestBody,
    retryFrom?: PreparedRun,
    retryParent: MessageId = null,
  ): Promise<PreparedRun> {
    const { modelType, publicModel } = resolveModel(body);
    const thinking = resolveThinking(body);
    const search = resolveSearch(body, modelType);
    const conversation = retryFrom ? null : this.sessions.resolve(body);
    const reusedSession = retryFrom
      ? false
      : Boolean(conversation?.sessionId && this.sessions.has(conversation.sessionId));
    const previous = !retryFrom && reusedSession && conversation?.sessionId
      ? this.sessions.get(conversation.sessionId)
      : undefined;
    const built = retryFrom
      ? null
      : buildDeepSeekPrompt(body, { reusedSession, ...(previous ? { previous } : {}) });
    const prompt = retryFrom
      ? buildToolRecoveryPrompt(body, retryFrom.requestTurns, retryFrom.latestUserText).trim()
      : built?.prompt.trim() ?? "";
    if (!prompt) throw new HttpError(400, "empty input");

    const auth = await this.login.dumpCurrent();
    const page = await this.login.page();
    const prepared = await prepareCompletion({
      page,
      powWorkerUrl: this.config.powWorkerUrl,
      modelType,
      fallbackToken: auth.token,
      // A retry stays in the original upstream session instead of orphaning it.
      sessionId: retryFrom
        ? retryFrom.sessionId
        : reusedSession
          ? conversation?.sessionId ?? null
          : null,
      reuseSession: retryFrom ? true : reusedSession,
    });
    const parentMessageId = retryFrom
      ? retryParent
      : reusedSession
        ? conversation?.parentMessageId ?? this.sessions.get(prepared.sessionId)?.lastResponseMessageId ?? null
        : null;
    const currentAuth = await this.login.dumpCurrent();
    const upstream = await openCompletionStream(this.config.baseUrl, currentAuth, {
      token: prepared.token,
      powHeader: prepared.powHeader,
      sessionId: prepared.sessionId,
      modelType,
      prompt,
      thinking,
      search,
      parentMessageId,
    });
    const retry = retryFrom ? retryFrom.retry + 1 : 0;
    this.logger.debug("DeepSeek completion prepared", {
      sessionId: prepared.sessionId,
      reused: reusedSession,
      parentMessageId,
      promptChars: prompt.length,
      retry,
      modelType,
      thinking,
      search,
    });
    const carried = retryFrom ?? built;
    return {
      upstream,
      prompt,
      sessionId: prepared.sessionId,
      parentMessageId,
      modelType,
      publicModel,
      thinking,
      search,
      reusedSession,
      convKey: retryFrom
        ? retryFrom.convKey
        : conversation?.key ?? conversation?.pendingFingerprint ?? null,
      requestTurns: carried?.requestTurns ?? [],
      allTurns: carried?.allTurns ?? [],
      instructionFingerprint: carried?.instructionFingerprint ?? "",
      toolsFingerprint: carried?.toolsFingerprint ?? "", latestUserText: carried?.latestUserText ?? "",
      hasTools: carried?.hasTools ?? false, toolHints: carried?.toolHints ?? [],
      retry,
    };
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
      sessionId: run.sessionId,
      modelType: run.modelType,
      responseMessageId,
      convKey: run.convKey,
      fullTurns,
      assistantContent: responseText,
      instructionFingerprint: run.instructionFingerprint,
      toolsFingerprint: run.toolsFingerprint,
    });
  }
}
