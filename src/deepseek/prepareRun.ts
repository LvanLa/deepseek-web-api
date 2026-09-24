/** Resolves session reuse, PoW, and the upstream stream for one completion. */
import type { LoginManager } from "../browser/login.js";
import type { AppConfig } from "../config/env.js";
import { HttpError } from "../utils/errors.js";
import type { Logger } from "../utils/logger.js";
import type { ChatRun } from "./chatStream.js";
import { openCompletionStream } from "./completion.js";
import { resolveModel, resolveSearch, resolveThinking } from "./models.js";
import { prepareCompletion } from "./pow.js";
import { buildDeepSeekPrompt, buildToolRecoveryPrompt } from "./promptBuild.js";
import type { ConversationResolution, MessageId, SessionStore } from "./sessionStore.js";
import type { MessageTurn, ModelType, RequestBody } from "./types.js";

export interface PreparedRun extends ChatRun {
  prompt: string; parentMessageId: MessageId; modelType: ModelType;
  thinking: boolean; search: boolean; reusedSession: boolean;
  convKey: string | null; requestTurns: MessageTurn[]; allTurns: MessageTurn[];
  instructionFingerprint: string; toolsFingerprint: string;
  latestUserText: string; hasTools: boolean; retry: number;
}

export interface PrepareDeps {
  config: AppConfig; login: LoginManager; sessions: SessionStore; logger: Logger;
}

/** Build the initial request or a compact retry (optionally in a fresh session). */
export async function prepareRun(
  deps: PrepareDeps,
  body: RequestBody,
  retryFrom?: PreparedRun,
  retryParent: MessageId = null,
  freshSession = false,
): Promise<PreparedRun> {
  const { config, login, sessions, logger } = deps;
  const { modelType, publicModel } = resolveModel(body);
  const thinking = resolveThinking(body);
  const search = resolveSearch(body, modelType);
  let resolution: ConversationResolution | null = retryFrom ? null : sessions.resolve(body);
  if (!retryFrom && resolution && !resolution.sessionId && resolution.key) {
    // Sliding-window agents do not replay cumulative history; bind to the
    // latest session reusing identical instructions instead of forking.
    const probe = buildDeepSeekPrompt(body, { reusedSession: false });
    const sticky = probe.instructionFingerprint
      ? sessions.resolveInstruction(probe.instructionFingerprint)
      : undefined;
    if (sticky) resolution = { ...sticky, key: resolution.key };
  }
  const conversation = resolution;
  const reusedSession = retryFrom
    ? false
    : Boolean(conversation?.sessionId && sessions.has(conversation.sessionId));
  const previous = !retryFrom && reusedSession && conversation?.sessionId
    ? sessions.get(conversation.sessionId)
    : undefined;
  const built = retryFrom
    ? null
    : buildDeepSeekPrompt(body, { reusedSession, ...(previous ? { previous } : {}) });
  const prompt = retryFrom
    ? buildToolRecoveryPrompt(body, retryFrom.requestTurns, retryFrom.latestUserText).trim()
    : built?.prompt.trim() ?? "";
  if (!prompt) throw new HttpError(400, "empty input");
  // The web silently rejects oversized turns; warn before spending a retry.
  if (prompt.length > 130000) logger.warn(
    "Prompt exceeds 130000 chars; DeepSeek web may return an empty stream",
    { promptChars: prompt.length, reusedSession: !retryFrom && reusedSession },
  );

  const auth = await login.dumpCurrent();
  const page = await login.page();
  const prepared = await prepareCompletion({
    page,
    powWorkerUrl: config.powWorkerUrl,
    modelType,
    fallbackToken: auth.token,
    // A retry normally keeps the original session; a fresh-session retry
    // after overflow creates a new one instead.
    sessionId: retryFrom
      ? (freshSession ? null : retryFrom.sessionId)
      : (reusedSession ? (conversation?.sessionId ?? null) : null),
    reuseSession: retryFrom ? !freshSession : reusedSession,
  });
  const parentMessageId = retryFrom
    ? (freshSession ? null : retryParent)
    : (reusedSession
      ? (conversation?.parentMessageId ?? sessions.get(prepared.sessionId)?.lastResponseMessageId ?? null)
      : null);
  const currentAuth = await login.dumpCurrent();
  const upstream = await openCompletionStream(config.baseUrl, currentAuth, {
    token: prepared.token, powHeader: prepared.powHeader, sessionId: prepared.sessionId,
    modelType, prompt, thinking, search, parentMessageId,
  });
  const retry = retryFrom ? retryFrom.retry + 1 : 0;
  logger.debug("DeepSeek completion prepared", {
    sessionId: prepared.sessionId, reused: reusedSession, parentMessageId,
    promptChars: prompt.length, retry, modelType, thinking, search,
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
