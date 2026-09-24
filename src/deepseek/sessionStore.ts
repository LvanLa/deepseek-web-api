/** Persists DeepSeek session lineage; identity is model-agnostic to prevent forking. */
import type { Logger } from "../utils/logger.js";
import { isRecord } from "../utils/json.js";
import { normalizeText } from "../utils/text.js";
import { requestConversationTurns } from "./promptBuild.js";
import { stripBareWrapperTags } from "./dsmlToolCalls.js";
import { SessionFile } from "./sessionFile.js";
import { fingerprint, fpKey, foldTurns, turnsEqual, turnsPrefix, turnsSuffix } from "./sessionTurns.js";
import type { MessageTurn, ModelType, RequestBody } from "./types.js";
const MAX_SESSIONS = 500;
const MAX_TURNS = 40;
const HASHED_FP_KEY = /^fp:[0-9a-f]{64}$/;

/** Last at most ``cap`` turns, starting at a user boundary when reachable. */
function requestWindow(turns: readonly MessageTurn[], cap: number): MessageTurn[] {
  if (turns.length <= cap) return turns.map((turn) => ({ ...turn }));
  let start = turns.length - cap;
  while (start < turns.length && turns[start]?.role !== "user") start += 1;
  if (start >= turns.length) start = turns.length - cap;
  return turns.slice(start).map((turn) => ({ ...turn }));
}
export type MessageId = string | number | null;
export interface SessionEntry {
  lastResponseMessageId: MessageId;
  lastModelType?: ModelType; modelType?: ModelType;
  instructionFingerprint?: string; toolsFingerprint?: string;
  updatedAt: number; turns: MessageTurn[];
}
export interface ConversationResolution {
  sessionId: string | null; parentMessageId: MessageId; key: string | null;
  pendingFingerprint?: string; createIfMissing?: boolean;
}
/** Exclude the trailing request turn because it is not stored history yet. */
const historyTurns = (messages: unknown): MessageTurn[] => {
  const turns = requestConversationTurns({ messages });
  return turns.at(-1)?.role === "assistant" ? turns : turns.slice(0, -1);
};
function metadata(body: RequestBody): Record<string, unknown> {
  return isRecord(body.metadata) ? body.metadata : {};
}

function stringId(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
function parentId(value: unknown): MessageId {
  return typeof value === "string" || typeof value === "number" ? value : null;
}

function parseModelType(value: unknown): ModelType | undefined {
  return value === "default" || value === "expert" ? value : undefined;
}
function parseSessionEntry(value: unknown): SessionEntry | null {
  if (!isRecord(value) || !Array.isArray(value.turns) || typeof value.updatedAt !== "number") return null;
  const turns: MessageTurn[] = [];
  for (const turn of value.turns) {
    if (!isRecord(turn) || typeof turn.role !== "string" || typeof turn.content !== "string") return null;
    // Legacy disk entries may contain wrapper leftovers or standalone residue.
    const content = turn.role === "assistant" ? stripBareWrapperTags(turn.content) : turn.content;
    if (content.trim()) turns.push({ role: turn.role, content });
  }
  const lastResponseMessageId = parentId(value.lastResponseMessageId);
  const lastModelType = parseModelType(value.lastModelType);
  const modelType = parseModelType(value.modelType);
  const instructionFingerprint = stringId(value.instructionFingerprint) ?? undefined;
  const toolsFingerprint = stringId(value.toolsFingerprint) ?? undefined;
  const optional: Partial<SessionEntry> = {};
  if (lastModelType) optional.lastModelType = lastModelType;
  if (modelType) optional.modelType = modelType;
  if (instructionFingerprint) optional.instructionFingerprint = instructionFingerprint;
  if (toolsFingerprint) optional.toolsFingerprint = toolsFingerprint;
  return { lastResponseMessageId, updatedAt: value.updatedAt, turns, ...optional };
}

/** Public session index used by every completion path. */
export class SessionStore {
  private readonly sessions = new Map<string, SessionEntry>();
  private readonly convIndex = new Map<string, string>();
  private readonly instructionIndex = new Map<string, string>();
  private readonly storage: SessionFile;
  private readonly logger: Logger | undefined;

  constructor(file?: string, logger?: Logger) {
    this.logger = logger;
    this.storage = new SessionFile(file, logger);
    this.load();
  }
  /** Flush buffered persistence and release the single-instance lock. */
  close(): void {
    this.storage.close();
  }

  has(sessionId: string): boolean {
    return this.sessions.has(sessionId);
  }
  get(sessionId: string): SessionEntry | undefined {
    return this.sessions.get(sessionId);
  }

  /** Sticky fallback: bind to the latest session reusing identical instructions. */
  resolveInstruction(fp: string): ConversationResolution | undefined {
    const sessionId = this.instructionIndex.get(fp);
    if (!sessionId || !this.sessions.has(sessionId)) return undefined;
    return this.found(sessionId, `instr:${fp.slice(0, 16)}`);
  }

  /** Resolve explicit IDs first, then previous response IDs, then history fingerprints. */
  resolve(body: RequestBody): ConversationResolution {
    const meta = metadata(body);
    const explicit = stringId(
      body.chat_session_id ?? body.conversation ?? body.conversation_id
        ?? meta.chat_session_id ?? meta.conversation_id,
    );
    if (explicit) return this.resolveExplicit(explicit, parentId(body.parent_message_id), "id");

    const previous = stringId(
      body.previous_response_id ?? body.previous_response ?? meta.previous_response_id,
    );
    if (previous) {
      const match = /^resp_([0-9a-f-]{36})/i.exec(previous) ?? /^resp_(.+)$/.exec(previous);
      const sessionId = match?.[1];
      if (sessionId) return this.resolveExplicit(sessionId, parentId(body.parent_message_id), "prev");
    }

    const messages = this.requestMessages(body);
    if (messages && messages.length > 1) return this.resolveHistory(messages);
    return { sessionId: null, parentMessageId: null, key: null };
  }

  /**
   * Save the response message ID required as the next parent_message_id.
   * The stored turns are the client's expanded request window (user, assistant
   * call turns, tool results) plus the structured assistant response, so the
   * next replayed request matches stored turns by construction.
   */
  remember(input: {
    sessionId: string; modelType: ModelType; responseMessageId: MessageId;
    convKey?: string | null; fullTurns: MessageTurn[]; assistantContent: string;
    instructionFingerprint?: string; toolsFingerprint?: string;
  }): void {
    const assistantText = normalizeText(input.assistantContent);
    if (!assistantText.trim()) {
      this.logger?.warn("Skipping empty assistant turn", { sessionId: input.sessionId, responseMessageId: input.responseMessageId });
      return;
    }
    const previous = this.sessions.get(input.sessionId);
    const turns: MessageTurn[] = [
      ...requestWindow(input.fullTurns, MAX_TURNS - 1),
      { role: "assistant", content: assistantText },
    ];
    const instructionFingerprint = input.instructionFingerprint ?? previous?.instructionFingerprint;
    const toolsFingerprint = input.toolsFingerprint ?? previous?.toolsFingerprint;
    const entry: SessionEntry = {
      lastResponseMessageId: input.responseMessageId ?? previous?.lastResponseMessageId ?? null,
      lastModelType: input.modelType,
      modelType: input.modelType,
      ...(instructionFingerprint ? { instructionFingerprint } : {}),
      ...(toolsFingerprint ? { toolsFingerprint } : {}),
      updatedAt: Date.now(),
      turns,
    };
    this.sessions.set(input.sessionId, entry);
    this.prune();
    if (input.convKey) this.convIndex.set(input.convKey, input.sessionId);
    if (instructionFingerprint) this.instructionIndex.set(instructionFingerprint, input.sessionId);
    // Index folded-round fingerprints at each assistant boundary so any
    // completed logical turn can resume independently.
    const folded = foldTurns(turns);
    this.convIndex.set(fpKey(fingerprint(folded)), input.sessionId);
    for (let end = 1; end <= folded.length; end += 1) {
      if (folded[end - 1]?.role === "assistant")
        this.convIndex.set(fpKey(fingerprint(folded.slice(0, end))), input.sessionId);
    }
    this.save();
  }
  private resolveExplicit(sessionId: string, fallbackParent: MessageId, prefix: "id" | "prev"): ConversationResolution {
    const entry = this.sessions.get(sessionId);
    return {
      sessionId, parentMessageId: entry?.lastResponseMessageId ?? fallbackParent,
      key: `${prefix}:${sessionId}`, ...(!entry ? { createIfMissing: true } : {}),
    };
  }

  private requestMessages(body: RequestBody): unknown[] | null {
    if (Array.isArray(body.messages)) return body.messages;
    if (Array.isArray(body.input) && body.input.some((item) =>
      isRecord(item) && (item.role || ["message", "function_call", "function_call_output"].includes(String(item.type))),
    )) {
      return body.input;
    }
    return null;
  }

  /** Match folded rounds via exact fingerprints, then equality, prefix, user chain, assistant tail. */
  private resolveHistory(messages: unknown[]): ConversationResolution {
    const turns = historyTurns(messages);
    if (turns.length === 0) return { sessionId: null, parentMessageId: null, key: null };
    const folded = foldTurns(turns);
    const key = fpKey(fingerprint(folded));
    const exact = this.convIndex.get(key) ?? this.findLegacy(fingerprint(folded));
    if (exact && this.sessions.has(exact)) return this.found(exact, key);

    let best: { sessionId: string; score: number } | undefined;
    for (const [sessionId, entry] of this.sessions) {
      const score = this.historyScore(folded, foldTurns(entry.turns), turns);
      if (score > 0 && (!best || score > best.score)) best = { sessionId, score };
    }
    if (best) return this.found(best.sessionId, key);
    return { sessionId: null, parentMessageId: null, key, pendingFingerprint: key };
  }

  private historyScore(incoming: MessageTurn[], storedFolded: MessageTurn[], rawIncoming: MessageTurn[]): number {
    if (
      turnsEqual(storedFolded, incoming) ||
      turnsPrefix(storedFolded, incoming) ||
      turnsPrefix(incoming, storedFolded) ||
      turnsSuffix(storedFolded, incoming) ||
      turnsSuffix(incoming, storedFolded)
    ) {
      return Math.min(storedFolded.length, incoming.length);
    }
    // Logical user chain identical: tolerate minor tool-block formatting drift.
    const usersOf = (values: MessageTurn[]): string[] =>
      values.filter((turn) => turn.role === "user").map((turn) => normalizeText(turn.content));
    const usersIn = usersOf(incoming);
    const usersStored = usersOf(storedFolded);
    if (
      usersIn.length > 0 &&
      usersIn.length === usersStored.length &&
      usersIn.every((user, index) => user === usersStored[index])
    ) {
      return usersIn.length + 0.25;
    }
    const lastIncoming = [...rawIncoming].reverse().find((turn) => turn.role === "assistant");
    const lastStored = [...storedFolded].reverse().find((turn) => turn.role === "assistant");
    return lastIncoming?.content &&
      normalizeText(lastIncoming.content) === normalizeText(lastStored?.content ?? "")
      ? 0.5
      : 0;
  }

  private found(sessionId: string, key: string): ConversationResolution {
    return {
      sessionId,
      parentMessageId: this.sessions.get(sessionId)?.lastResponseMessageId ?? null,
      key,
    };
  }

  private findLegacy(value: string): string | undefined {
    return ["default", "expert", "vision"]
      .map((model) => this.convIndex.get(fpKey(`${model}:${value}`)))
      .find((sessionId) => sessionId !== undefined);
  }
  private prune(): void {
    if (this.sessions.size <= MAX_SESSIONS) return;
    const oldest = [...this.sessions.entries()].sort((a, b) => a[1].updatedAt - b[1].updatedAt);
    for (const [id] of oldest.slice(0, this.sessions.size - MAX_SESSIONS)) this.sessions.delete(id);
  }
  private load(): void {
    const raw = this.storage.read();
    if (!raw) return;
    try {
      const parsed: unknown = JSON.parse(raw);
      if (!isRecord(parsed) || !isRecord(parsed.sessions) || !isRecord(parsed.convs)) return;
      for (const [id, value] of Object.entries(parsed.sessions)) {
        const entry = parseSessionEntry(value);
        if (entry) this.sessions.set(id, entry);
      }
      for (const [key, value] of Object.entries(parsed.convs)) {
        if (typeof value !== "string" || !this.sessions.has(value)) continue;
        // Rewrite raw legacy fingerprint keys to hashed form; keep explicit keys intact.
        const indexed = key.startsWith("fp:") && !HASHED_FP_KEY.test(key) ? fpKey(key.slice(3)) : key;
        this.convIndex.set(indexed, value);
      }
      if (isRecord(parsed.instructions)) {
        for (const [fp, id] of Object.entries(parsed.instructions)) {
          if (typeof id === "string" && this.sessions.has(id)) this.instructionIndex.set(fp, id);
        }
      }
    } catch (error) {
      this.logger?.warn("could not read sessions file; starting with an empty index", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private save(): void {
    const convs: Record<string, string> = {};
    for (const [key, sessionId] of this.convIndex) {
      if (this.sessions.has(sessionId)) convs[key] = sessionId;
    }
    const instructions: Record<string, string> = {};
    for (const [fp, sessionId] of this.instructionIndex) {
      if (this.sessions.has(sessionId)) instructions[fp] = sessionId;
    }
    this.storage.requestSave(JSON.stringify({
      sessions: Object.fromEntries(this.sessions), convs, instructions,
    }));
  }
}
