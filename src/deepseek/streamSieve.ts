/** Incrementally separates visible prose from complete tool-call blocks. */
import { parseToolCalls, type OpenAIToolCall, type ParsedToolCalls, type ToolDefHint } from "./toolCalls.js";

export type SieveEvent =
  | { type: "text"; delta: string }
  | { type: "toolCalls"; calls: OpenAIToolCall[]; content: string };

const MAX_CAPTURE_BUFFER = 1024 * 1024;

const LOOSE_OPEN = /<\s*(?:tool[_-]?call|_?call)\b[^>]*>/gi;
const LOOSE_CLOSE = /<\s*\/\s*(?:tool[_-]?call|_?call)\b[^>]*>/gi;
const DSML_OPEN = /<[｜]{1,2}DSML[｜]{1,2}[ \t]*(invoke|calls|tool_calls|function_calls)\b[^>]*>/gi;
const DSML_CLOSE = /<\/[｜]{1,2}DSML[｜]{1,2}[ \t]*(invoke|calls|tool_calls|function_calls)\b[^>]*>/gi;
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
  const indexes = [LOOSE_OPEN, DSML_OPEN, BARE_WRAPPER_OPEN].map((pattern) => {
    pattern.lastIndex = 0;
    return pattern.exec(text)?.index ?? -1;
  }).filter((index) => index >= 0);
  return indexes.length ? Math.min(...indexes) : -1;
}

const OPENER_KEYWORDS = [
  "tool_call", "tool-call", "toolcall", "tool_calls",
  "call", "calls", "_call", "function_calls",
];
const DSML_KEYWORDS = ["invoke", "calls", "tool_calls", "function_calls"];

/** True while ``rest`` is still a prefix of ``keyword`` or its partial attrs. */
function keywordOrAttrs(keyword: string, rest: string, leadingSpace = false): boolean {
  const body = leadingSpace ? rest.replace(/^\s+/, "") : rest;
  if (keyword.startsWith(body)) return true;
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${escaped}(?:\\s[^<>]*)?$`).test(body);
}

/** A trailing "<..." suffix that could still grow into a tool-call opener. */
function couldStartTag(tail: string): boolean {
  if (!tail.startsWith("<")) return false;
  const rest = tail.slice(1).replace(/^\s+/, "").toLowerCase();
  if (!rest) return true;
  if (rest[0] === "｜") {
    // Bars, DSML, a second bar group, optional whitespace, and the keyword or
    // attrs may each arrive in their own network chunk; hold any such prefix.
    const head = rest.match(/^｜{1,2}dsml(?:｜{1,2})?/)?.[0];
    if (!head) return /^｜{0,2}(?:d|ds|dsm|dsml)?$/.test(rest);
    return DSML_KEYWORDS.some((word) => keywordOrAttrs(word, rest.slice(head.length), true));
  }
  return OPENER_KEYWORDS.some((keyword) => keywordOrAttrs(keyword, rest));
}

/** Index just past a balanced top-level JSON object starting at ``start``, or -1. */
function jsonObjectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index] ?? "";
    if (inString) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') inString = false;
    } else if (char === '"') inString = true;
    else if (char === "{") depth += 1;
    else if (char === "}" && --depth === 0) return index + 1;
  }
  return -1;
}

/** Feed output deltas one chunk at a time, holding back only potential tool-call suffixes. */
export class StreamSieve {
  readonly fullOutput: string[] = [];
  private readonly emittedCalls: OpenAIToolCall[] = [];
  private pending = "";
  private capture = "";
  private captureKind: CaptureKind | null = null;
  private nonWhitespaceReleased = false;

  constructor(private readonly seed = "tool", private readonly toolHints: readonly ToolDefHint[] = []) {}

  /** Tool calls already committed by this sieve, in emission order. */
  get calls(): OpenAIToolCall[] { return this.emittedCalls; }

  feed(chunk: string): SieveEvent[] {
    if (!chunk) return [];
    this.fullOutput.push(chunk);
    const events: SieveEvent[] = [];
    if (this.captureKind !== null) {
      this.capture += chunk;
      if (this.releaseOversized(events)) return events;
      const jsonEnd = this.captureKind === "json" ? jsonObjectEnd(this.capture, 0) : -1;
      if (this.captureKind === "tag" ? this.captureComplete() : jsonEnd >= 0) {
        const committed = this.captureKind === "tag"
          ? this.commitTagCapture()
          : this.commitJsonCapture(jsonEnd);
        events.push(...committed, ...this.drainPending());
      }
      return events;
    }
    this.pending += chunk;
    return this.drainPending();
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
      if (this.pending === snapshot) {
        this.addText(events, snapshot);
        this.pending = "";
      }
    }
    return events;
  }

  private drainPending(): SieveEvent[] {
    const events: SieveEvent[] = [];
    while (this.pending) {
      const opener = findOpener(this.pending);
      if (opener >= 0) {
        this.addText(events, this.pending.slice(0, opener));
        this.startCapture("tag", this.pending.slice(opener));
        if (this.releaseOversized(events)) return events;
        if (this.captureComplete()) events.push(...this.commitTagCapture());
        else return events;
      }
      if (!this.nonWhitespaceReleased) {
        const brace = this.pending.match(/^\s*\{/);
        if (brace) {
          const braceIndex = (brace.index ?? 0) + brace[0].length - 1;
          this.addText(events, this.pending.slice(0, braceIndex));
          this.startCapture("json", this.pending.slice(braceIndex));
          if (this.releaseOversized(events)) return events;
          const end = jsonObjectEnd(this.capture, 0);
          if (end >= 0) events.push(...this.commitJsonCapture(end));
          else return events;
        }
      }
      const [safe, hold] = this.splitSafe(this.pending);
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
        lastMatchEnd(BARE_WRAPPER_CLOSE, buffer));
      rest = consumedEnd > 0 ? buffer.slice(consumedEnd) : "";
      this.pushCommitted(events, parsed);
    } else if (parsed.content.trim()) {
      // Orphan close tags stay hidden; other closed prose is released.
      if (!/^\s*<\/[^>]+>\s*$/.test(parsed.content)) this.addText(events, buffer);
    }
    // Protocol-shaped garbage with no reparable call stays hidden; turn-end
    // resolution owns the empty-fallback/retry decision.
    // Hold a trailing partial tag fragment (no ">" yet) for the next chunk.
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
    // A leading JSON object that is not a protocol-only tool call: release it
    // and stop considering stream-start braces as tool-call candidates.
    this.resetCapture();
    this.pending = buffer;
    this.markReleased(head);
    return this.drainPending();
  }

  private captureComplete(): boolean {
    const buffer = this.capture;
    if (regexPresent(LOOSE_OPEN, buffer) && regexPresent(LOOSE_CLOSE, buffer)) return true;
    if (regexPresent(DSML_OPEN, buffer)) {
      // Once a wrapper is open, only its matching wrapper close completes the
      // capture; closing individual invokes must not commit the block early.
      for (const name of ["calls", "tool_calls", "function_calls"]) {
        const open = new RegExp(`<[｜]{1,2}DSML[｜]{1,2}[ \\t]*${name}\\b`, "i");
        if (open.test(buffer)) {
          const close = new RegExp(
            `(?:</[｜]{1,2}DSML[｜]{1,2}[ \\t]*${name}\\b[^>]*>|<\\s*\\/\\s*${name}\\s*>)`, "i");
          return close.test(buffer);
        }
      }
      // No wrapper: a loose standalone invoke completes when it is closed.
      return regexPresent(/<[｜]{1,2}DSML[｜]{1,2}[ \t]*invoke\b[^>]*>/i, buffer) &&
        regexPresent(/<\/[｜]{1,2}DSML[｜]{1,2}[ \t]*invoke\b[^>]*>/i, buffer);
    }
    if (regexPresent(BARE_WRAPPER_OPEN, buffer)) {
      if (regexPresent(BARE_WRAPPER_CLOSE, buffer)) return true;
      // A DSML-form wrapper close may survive an ASCII-mangled opener.
      if (regexPresent(
        /<\/[｜]{1,2}DSML[｜]{1,2}[ \t]*(calls|tool_calls|function_calls)\b[^>]*>/i, buffer)) return true;
      // Fully-mangled wrapper with no reliable close: settle once an invoke closes.
      return regexPresent(DSML_OPEN, buffer) &&
        regexPresent(/<\/[｜]{1,2}DSML[｜]{1,2}[ \t]*invoke\b[^>]*>/i, buffer);
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
    return couldStartTag(tail) ? [text.slice(0, lastLt), tail] : [text, ""];
  }

  private markReleased(text: string): void { if (/\S/.test(text)) this.nonWhitespaceReleased = true; }

  private resetCapture(): void { this.capture = ""; this.captureKind = null; }
}
