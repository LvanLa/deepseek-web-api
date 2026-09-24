/** Incrementally separates visible prose from complete tool-call blocks. */
import { parseToolCalls, type OpenAIToolCall, type ParsedToolCalls, type ToolDefHint } from "./toolCalls.js";
import {
  DSML_CLOSE,
  DSML_INVOKE_CLOSE,
  DSML_INVOKE_OPEN,
  DSML_OPEN,
  DSML_WRAPPER_CLOSE,
  DSML_WRAPPER_NAME_LIST,
  dsmlBlockClose,
  dsmlPrefix,
} from "./dsmlToolCalls.js";
import { XML_INVOKE_CLOSE, XML_INVOKE_OPEN } from "./xmlToolCalls.js";
import { RESULT_CLOSE, RESULT_OPEN } from "./toolResultTags.js";

export type SieveEvent = { type: "text"; delta: string }
  | { type: "toolCalls"; calls: OpenAIToolCall[]; content: string };

const MAX_CAPTURE_BUFFER = 1024 * 1024;

const LOOSE_OPEN = /<\s*(?:tool[_-]?call|_?call)\b[^>]*>/gi;
const LOOSE_CLOSE = /<\s*\/\s*(?:tool[_-]?call|_?call)\b[^>]*>/gi;
const BARE_WRAPPER_OPEN = /<\s*(calls|tool_calls|function_calls)\s*>/gi;
const BARE_WRAPPER_CLOSE = /<\s*\/\s*(calls|tool_calls|function_calls)\s*>/gi;

type CaptureKind = "tag" | "json";

function regexPresent(pattern: RegExp, text: string): boolean {
  pattern.lastIndex = 0;
  return pattern.test(text);
}

function lastMatchEnd(pattern: RegExp, text: string): number {
  pattern.lastIndex = 0;
  let end = 0;
  for (const match of text.matchAll(pattern)) end = (match.index ?? 0) + match[0].length;
  return end;
}

/** Earliest index where any tool-call opener could begin, or -1. */
function findOpener(text: string): number {
  const patterns = [LOOSE_OPEN, DSML_OPEN, BARE_WRAPPER_OPEN, XML_INVOKE_OPEN, RESULT_OPEN];
  const indexes = patterns.flatMap((pattern) => {
    pattern.lastIndex = 0;
    const index = pattern.exec(text)?.index ?? -1;
    return index >= 0 ? [index] : [];
  });
  return indexes.length ? Math.min(...indexes) : -1;
}

/** Length of leading orphan protocol close-tag residue, separators allowed, or 0. */
function orphanCloseLead(text: string): number {
  const tag = "</(?:[｜]{1,2}DSML[｜]{1,2}\\s*[a-z_]+|\\s*(?:calls|tool_calls|function_calls|tool[_-]?call|_?call|invoke|parameter|tool[ _-]?call[ _-]?result|call[ _-]?result|command_id|command_status|command_run_logs|process_id|terminal_id)\\b)[^>]*>";
  return text.match(new RegExp(`^[\\s,]*(?:${tag}[\\s,]*)+`))?.[0].length ?? 0;
}

const OPENER_KEYWORDS = ["tool_call", "tool-call", "toolcall", "tool_calls", "call", "calls", "_call", "function_calls"];
const DSML_KEYWORDS = ["invoke", "calls", "tool_calls", "function_calls"];
const XML_KEYWORDS = ["invoke", "parameter"];
const RESULT_KEYWORDS = ["tool_call_result", "toolcall_result", "tool-result", "call_result"];

/** True while ``rest`` is still a prefix of ``keyword`` or its partial attrs. */
function keywordOrAttrs(keyword: string, rest: string, leadingSpace = false): boolean {
  const body = leadingSpace ? rest.replace(/^\s+/, "") : rest;
  if (keyword.startsWith(body)) return true;
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${escaped}(?:\\s[^<>]*)?$`).test(body);
}

/** A trailing "<..." suffix that could still grow into an opener or closer. */
function couldStartTag(tail: string): boolean {
  if (!tail.startsWith("<")) return false;
  const rest = tail.slice(1).replace(/^\s+/, "").toLowerCase();
  if (!rest) return true;
  const closing = rest.startsWith("/");
  const body = rest.slice(closing ? 1 : 0);
  if (body[0] === "｜") {
    const head = body.match(/^｜{1,2}dsml(?:｜{1,2})?/)?.[0];
    if (!head) return /^｜{0,2}(?:d|ds|dsm|dsml)?$/.test(body);
    const after = body.slice(head.length).replace(/^\s+|>$/g, "");
    return after === "" || DSML_KEYWORDS.some((w) => keywordOrAttrs(w, after, true));
  }
  if (closing) return OPENER_KEYWORDS.some((k) => k.startsWith(body.replace(/>$/, ""))) ||
    OPENER_KEYWORDS.some((k) => new RegExp(`^${k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[^<>]*>$`).test(body)) ||
    XML_KEYWORDS.some((k) => keywordOrAttrs(k, body)) ||
    RESULT_KEYWORDS.some((k) => keywordOrAttrs(k, body));
  return OPENER_KEYWORDS.some((k) => keywordOrAttrs(k, body)) ||
    XML_KEYWORDS.some((k) => keywordOrAttrs(k, body)) ||
    RESULT_KEYWORDS.some((k) => keywordOrAttrs(k, body));
}

/** Index just past a balanced top-level JSON object starting at ``start``, or -1. */
function jsonObjectEnd(text: string, start: number): number {
  let depth = 0, inString = false, escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i]!;
    if (inString) { if (escaped) escaped = false; else if (char === "\\") escaped = true; else inString = char !== '"'; }
    else if (char === '"') inString = true;
    else if (char === "{") depth += 1;
    else if (char === "}" && --depth === 0) return i + 1;
  }
  return -1;
}

/** Feed output deltas one chunk at a time, holding back only potential tool-call suffixes. */
export class StreamSieve {
  readonly fullOutput: string[] = [];
  private readonly emittedCalls: OpenAIToolCall[] = [];
  private pending = ""; private capture = "";
  private captureKind: CaptureKind | null = null; private nonWhitespaceReleased = false;

  constructor(private readonly seed = "tool", private readonly toolHints: readonly ToolDefHint[] = []) {}
  /** Tool calls already committed by this sieve, in emission order. */
  get calls(): OpenAIToolCall[] { return this.emittedCalls; }

  feed(chunk: string): SieveEvent[] {
    if (!chunk) return [];
    this.fullOutput.push(chunk);
    if (this.captureKind === null) { this.pending += chunk; return this.drainPending(); }
    const events: SieveEvent[] = [];
    this.capture += chunk;
    if (this.releaseOversized(events)) return events;
    const jsonEnd = this.captureKind === "json" ? jsonObjectEnd(this.capture, 0) : -1;
    if (this.captureKind === "tag" ? this.captureComplete() : jsonEnd >= 0) {
      events.push(
        ...(this.captureKind === "tag" ? this.commitTagCapture() : this.commitJsonCapture(jsonEnd)),
        ...this.drainPending(),
      );
    }
    return events;
  }

  /**
   * Flush held buffers. A tag capture is parsed even without a close tag so a
   * truncated but recognizable call is recovered; other unclosed text is
   * released as prose and pure protocol garbage stays hidden.
   */
  flush(): SieveEvent[] {
    const events: SieveEvent[] = [];
    if (this.captureKind === "tag") events.push(...this.commitTagCapture());
    this.releaseCapture(events);
    while (this.pending) {
      const snapshot = this.pending;
      events.push(...this.drainPending());
      this.releaseCapture(events);
      if (this.pending === snapshot) { this.addText(events, snapshot); this.pending = ""; }
    }
    return events;
  }

  private drainPending(): SieveEvent[] {
    const events: SieveEvent[] = [];
    while (this.pending) {
      // Strip orphan close-tag residue from mismatched wrappers; never prose.
      const orphan = orphanCloseLead(this.pending);
      if (orphan) { this.pending = this.pending.slice(orphan); continue; }
      const opener = findOpener(this.pending);
      const emptyFollows = /^<\s*>\s*(?:\{|<\s*\/)/.test(this.pending);
      const startAt = opener >= 0 ? opener : emptyFollows ? 0 : -1;
      if (startAt >= 0) {
        const prefix = this.pending.slice(0, startAt);
        if (this.nonWhitespaceReleased || /[^\s,]/.test(prefix)) this.addText(events, prefix);
        this.startCapture("tag", this.pending.slice(startAt));
        if (this.releaseOversized(events) || !this.captureComplete()) return events;
        events.push(...this.commitTagCapture(), ...this.drainPending());
        return events;
      }
      const lead = !this.nonWhitespaceReleased ? this.pending.match(/^[\s,]*(?=\{)/) : null;
      if (lead) {
        this.startCapture("json", this.pending.slice(lead[0].length));
        if (this.releaseOversized(events)) return events;
        const end = jsonObjectEnd(this.capture, 0);
        if (end >= 0) events.push(...this.commitJsonCapture(end));
        return events;
      }
      const [safe, hold] = this.splitSafe(this.pending);
      if (!this.nonWhitespaceReleased && !/[^\s,]/.test(safe)) { if (hold) this.pending = safe + hold; return events; }
      this.addText(events, safe);
      this.pending = hold;
      return events;
    }
    return events;
  }

  private commitTagCapture(): SieveEvent[] {
    const buffer = this.capture;
    const parsed = parseToolCalls(buffer, this.seed, { tools: this.toolHints });
    const events: SieveEvent[] = [];
    let rest = "";
    if (parsed.toolCalls.length > 0) {
      const consumedEnd = Math.max(lastMatchEnd(LOOSE_CLOSE, buffer), lastMatchEnd(DSML_CLOSE, buffer),
        lastMatchEnd(BARE_WRAPPER_CLOSE, buffer), lastMatchEnd(XML_INVOKE_CLOSE, buffer),
        lastMatchEnd(RESULT_CLOSE, buffer));
      rest = consumedEnd > 0 ? buffer.slice(consumedEnd) : "";
      this.pushCommitted(events, parsed);
    } else if (parsed.content.trim() && !/^\s*<\/[^>]+>\s*$/.test(parsed.content)) {
      this.addText(events, buffer); // orphan close tags stay hidden; other prose is released
    }
    // Hold a partial tag fragment (no ">"); turn-end owns fallback/retry.
    if (rest && /^\s*<(?![\s\S]*>)/.test(rest)) {
      this.capture = rest;
      return events;
    }
    this.resetCapture();
    if (rest) this.pending = rest;
    return events;
  }

  private commitJsonCapture(end: number): SieveEvent[] {
    const buffer = this.capture;
    const head = buffer.slice(0, end);
    const rest = buffer.slice(end);
    const parsed = parseToolCalls(head, this.seed, { tools: this.toolHints });
    const events: SieveEvent[] = [];
    if (parsed.toolCalls.length > 0) {
      this.pushCommitted(events, parsed);
      this.resetCapture();
      if (rest) this.pending = rest;
      return events;
    }
    // Non-tool leading JSON: release it and stop treating braces as candidates.
    this.resetCapture();
    this.pending = buffer;
    this.markReleased(head);
    return this.drainPending();
  }

  private captureComplete(): boolean {
    const buffer = this.capture;
    // Empty opener: commit on loose close only; unclosed recovered at flush.
    if (/^<\s*>/.test(buffer)) return regexPresent(LOOSE_CLOSE, buffer);
    if (regexPresent(LOOSE_OPEN, buffer) && regexPresent(LOOSE_CLOSE, buffer)) return true;
    if (regexPresent(DSML_OPEN, buffer)) {
      // Only the matching wrapper close commits; invoke closes must not.
      for (const name of DSML_WRAPPER_NAME_LIST) {
        if (dsmlPrefix(name).test(buffer)) return dsmlBlockClose(name).test(buffer);
      }
      // A loose standalone invoke completes when it is closed.
      return regexPresent(DSML_INVOKE_OPEN, buffer) && regexPresent(DSML_INVOKE_CLOSE, buffer);
    }
    if (regexPresent(XML_INVOKE_OPEN, buffer)) return regexPresent(XML_INVOKE_CLOSE, buffer);
    if (regexPresent(RESULT_OPEN, buffer)) return regexPresent(RESULT_CLOSE, buffer);
    if (regexPresent(BARE_WRAPPER_OPEN, buffer)) {
      if (regexPresent(BARE_WRAPPER_CLOSE, buffer) || regexPresent(DSML_WRAPPER_CLOSE, buffer)) return true;
      return regexPresent(DSML_OPEN, buffer) && regexPresent(DSML_INVOKE_CLOSE, buffer);
    }
    // A continued orphan wrapper closer from a held rest completes capture.
    return regexPresent(DSML_CLOSE, buffer) && !regexPresent(LOOSE_OPEN, buffer) &&
      !regexPresent(DSML_OPEN, buffer) && !regexPresent(BARE_WRAPPER_OPEN, buffer);
  }

  /** Emit a text delta and remember that non-whitespace prose was released. */
  private addText(events: SieveEvent[], text: string): void {
    if (text) events.push({ type: "text", delta: text });
    this.markReleased(text);
  }

  /** Deduplicate against committed calls, then emit fresh calls and their content. */
  private pushCommitted(events: SieveEvent[], parsed: ParsedToolCalls): void {
    const isFresh = (call: OpenAIToolCall): boolean => !this.emittedCalls.some((seen) =>
      seen.function.name === call.function.name && seen.function.arguments === call.function.arguments);
    const fresh = parsed.toolCalls.filter(isFresh);
    if (fresh.length === 0) return;
    if (parsed.content) events.push({ type: "text", delta: parsed.content });
    events.push({ type: "toolCalls", calls: fresh, content: parsed.content });
    this.emittedCalls.push(...fresh);
  }

  /** Move ``text`` into capture, leaving no pending suffix. */
  private startCapture(kind: CaptureKind, text: string): void {
    this.capture = text; this.captureKind = kind; this.pending = "";
  }

  /** If capture exceeds the size cap, force it out as prose and return true. */
  private releaseOversized(events: SieveEvent[]): boolean {
    if (this.capture.length <= MAX_CAPTURE_BUFFER) return false;
    this.addText(events, this.capture);
    this.resetCapture();
    return true;
  }

  /** Release an active capture buffer as plain text (used by flush). */
  private releaseCapture(events: SieveEvent[]): void {
    if (this.captureKind === null) return;
    this.addText(events, this.capture);
    this.resetCapture();
  }

  private splitSafe(text: string): [string, string] {
    const lastLt = text.lastIndexOf("<");
    if (lastLt < 0) return [text, ""];
    const tail = text.slice(lastLt);
    if (couldStartTag(tail) || /^<\s*>$/.test(tail)) return [text.slice(0, lastLt), tail];
    return [text, ""];
  }

  private markReleased(text: string): void { if (/[^\s,]/.test(text)) this.nonWhitespaceReleased = true; }

  private resetCapture(): void { this.capture = ""; this.captureKind = null; }
}
