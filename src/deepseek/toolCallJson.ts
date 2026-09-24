/** Repairs and locates JSON objects emitted by the text tool protocol. */
import { disambiguatePathEscapes } from "./pathEscapes.js";

export interface JsonObjectCandidate {
  start: number;
  end: number;
  raw: string;
  value: unknown | null;
}

const TOOL_FIRST_KEY = /^\{\s*"(?:name|tool|arguments|function)"\s*:/;
const MAX_MISSING_CLOSERS = 8;

function parseObject(text: string): unknown | null {
  try {
    const value: unknown = JSON.parse(text);
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * Parse JSON tolerating single backslashes in Windows paths (for example
 * "f:\workspace\f"). Only attempted after a strict failure when a drive
 * prefix is present; doubling every backslash restores literal separators.
 */
function lenientObject(text: string): unknown | null {
  if (!/[A-Za-z]:[\\/]/.test(text)) return null;
  return parseObject(text.replace(/\\/g, "\\\\"));
}

function scanClosers(text: string): { end: number | null; missing: string[] | null } {
  const closers: string[] = [];
  let inString = false;
  let escape = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escape) escape = false;
      else if (char === "\\") escape = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') inString = true;
    else if (char === "{") closers.push("}");
    else if (char === "[") closers.push("]");
    else if (char === "}" || char === "]") {
      if (closers.at(-1) !== char) return { end: null, missing: null };
      closers.pop();
      if (closers.length === 0) return { end: index + 1, missing: [] };
    }
  }
  if (inString || escape) return { end: null, missing: null };
  return { end: null, missing: closers.reverse() };
}

function withoutTrailingCommas(text: string): string {
  return text.replace(/,\s*([}\]])/g, "$1").replace(/,\s*$/, "");
}

/** All tolerant string ends: quotes whose next non-space char is a follower. */
function greedyEnds(text: string, start: number, followers: string): number[] {
  const ends: number[] = [];
  for (let index = start; index < text.length; index += 1) {
    if (text[index] !== '"' || text[index - 1] === "\\") continue;
    let look = index + 1;
    while (look < text.length && /\s/.test(text[look]!)) look += 1;
    if (followers.includes(text[look] ?? "")) ends.push(index + 1);
  }
  // Shortest first: extend a string only when the tight end fails to parse.
  return ends;
}

interface GreedyResult { value: unknown; end: number }
type GreedyAccept = (result: GreedyResult) => GreedyResult | null;

const skipSpaces = (text: string, pos: number): number => {
  let index = pos;
  while (/\s/.test(text[index]!)) index += 1;
  return index;
};

function snapshot(built: unknown): unknown {
  return Array.isArray(built) ? built.slice() : { ...(built as Record<string, unknown>) };
}

function restore(built: unknown, shot: unknown): void {
  if (Array.isArray(built)) {
    built.length = 0;
    built.push(...(shot as unknown[]));
  } else {
    const record = built as Record<string, unknown>;
    for (const key of Object.keys(record)) delete record[key];
    Object.assign(record, shot);
  }
}

/** Continue after a value: comma starts another entry; close ends the container. */
function afterValue(text: string, pos: number, built: unknown, closer: string, accept: GreedyAccept): GreedyResult | null {
  const next = skipSpaces(text, pos);
  if (text[next] === ",") return parseEntry(text, next + 1, built, closer, accept);
  return text[next] === closer ? accept({ value: built, end: next + 1 }) : null;
}

function parseObjectEntry(text: string, index: number, built: unknown, closer: string, accept: GreedyAccept): GreedyResult | null {
  const record = built as Record<string, unknown>;
  if (text[index] !== '"') return null;
  for (const keyEnd of greedyEnds(text, index + 1, ":")) {
    const colon = skipSpaces(text, keyEnd);
    if (text[colon] !== ":") continue;
    const key = text.slice(index + 1, keyEnd - 1);
    // A real key never spans structure: long spans are backtrack artifacts.
    if (!key || /[:{}[\]]/.test(key)) continue;
    const previous = record[key];
    const taken = greedyValue(text, colon + 1, (child) => {
      record[key] = child.value;
      return afterValue(text, child.end, built, closer, accept);
    });
    if (taken) return taken;
    if (previous === undefined) delete record[key]; else record[key] = previous;
  }
  return null;
}

function parseArrayEntry(text: string, index: number, built: unknown, closer: string, accept: GreedyAccept): GreedyResult | null {
  const array = built as unknown[];
  const sizeBefore = array.length;
  const taken = greedyValue(text, index, (child) => {
    array.push(child.value);
    return afterValue(text, child.end, built, closer, accept);
  });
  if (!taken) array.length = sizeBefore;
  return taken;
}

function parseEntry(text: string, index: number, built: unknown, closer: string, accept: GreedyAccept): GreedyResult | null {
  index = skipSpaces(text, index);
  if (text[index] === closer) return accept({ value: built, end: index + 1 });
  const shot = snapshot(built);
  const result = closer === "}"
    ? parseObjectEntry(text, index, built, closer, accept)
    : parseArrayEntry(text, index, built, closer, accept);
  if (!result) restore(built, shot);
  return result;
}

/** Parse one value; accept is tried for every candidate end via backtracking. */
function greedyValue(text: string, at: number, accept: GreedyAccept): GreedyResult | null {
  const index = skipSpaces(text, at);
  const open = text[index];
  if (open === '"') {
    for (const end of greedyEnds(text, index + 1, ",}]")) {
      const taken = accept({ value: text.slice(index + 1, end - 1), end });
      if (taken) return taken;
    }
    return null;
  }
  if (open === "{" || open === "[") {
    const built: unknown = open === "{" ? {} : [];
    return parseEntry(text, index + 1, built, open === "{" ? "}" : "]", accept);
  }
  const literal = text.slice(index).match(/^(?:true|false|null)\b/)?.[0];
  if (literal) return accept({ value: JSON.parse(literal), end: index + literal.length });
  const number = text.slice(index).match(/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/)?.[0];
  return number ? accept({ value: Number(number), end: index + number.length }) : null;
}

const MAX_GREEDY_PARSES = 32;

/** Score parse structure: ambiguous parses must keep maximal structure. */
function structureScore(value: unknown): number {
  if (Array.isArray(value)) {
    return 1 + value.reduce((sum: number, item) => sum + structureScore(item), 0);
  }
  if (value && typeof value === "object") {
    return 2 + Object.values(value).reduce(
      (sum: number, child) => sum + 2 + structureScore(child), 0);
  }
  return 0;
}

/** Last-resort parse for unescaped quotes; every full consumption is ranked. */
function greedyObject(text: string): unknown | null {
  const parses: unknown[] = [];
  greedyValue(text, 0, (candidate) => {
    if (candidate.end === text.length) {
      parses.push(structuredClone(candidate.value));
      if (parses.length >= MAX_GREEDY_PARSES) return candidate;
    }
    return null;
  });
  if (parses.length === 0) return null;
  parses.sort((left, right) => structureScore(right) - structureScore(left));
  const best = parses[0];
  return best && typeof best === "object" && !Array.isArray(best) ? best : null;
}

/** Parse strictly; on a drive-prefixed payload, prefer path-escape disambiguation. */
function strictOrPathObject(text: string): unknown | null {
  const strict = parseObject(text);
  if (strict && /[A-Za-z]:[\\/]/.test(text)) {
    const fixed = parseObject(disambiguatePathEscapes(text));
    if (fixed && JSON.stringify(fixed) !== JSON.stringify(strict)) return fixed;
  }
  return strict ?? lenientObject(text);
}

/** Repair only structurally truncated objects; never guess unfinished string contents. */
export function repairJson(raw: string): unknown | null {
  const text = raw.trim();
  if (!text.startsWith("{")) return null;
  const direct = strictOrPathObject(text);
  if (direct) return direct;
  const scan = scanClosers(text);
  if (scan.end !== null) {
    const complete = withoutTrailingCommas(text.slice(0, scan.end));
    const parsed = strictOrPathObject(complete);
    if (parsed) return parsed;
  }
  if (scan.missing && scan.missing.length > 0 && scan.missing.length <= MAX_MISSING_CLOSERS) {
    const repaired = withoutTrailingCommas(text) + scan.missing.join("");
    const parsed = parseObject(repaired) ?? lenientObject(repaired);
    if (parsed) return parsed;
  }
  return greedyObject(text);
}

/** A brace is collectible at a line start or right after another object. */
function candidateStarts(text: string, allowAnyFirstObject: boolean): number[] {
  const starts: number[] = [];
  const firstObject = text.indexOf("{");
  let lastEnd = -1;
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "{") continue;
    const lineStart = text.lastIndexOf("\n", index - 1) + 1;
    // Separators are whitespace or commas between top-level bare objects.
    const lineGap = !/[^\s,]/.test(text.slice(lineStart, index));
    const objectGap = lastEnd >= 0 && index >= lastEnd &&
      !/[^\s,]/.test(text.slice(lastEnd, index));
    if (!lineGap && !objectGap) continue;
    if (!TOOL_FIRST_KEY.test(text.slice(index)) &&
        !(allowAnyFirstObject && index === firstObject)) continue;
    starts.push(index);
    const end = scanClosers(text.slice(index)).end;
    lastEnd = end !== null ? index + end : -1;
  }
  return starts;
}

/** Locate complete or repairable standalone tool-shaped JSON objects in one text segment. */
export function collectJsonObjects(
  text: string,
  allowAnyFirstObject = false,
): JsonObjectCandidate[] {
  const starts = candidateStarts(text, allowAnyFirstObject);
  const candidates: JsonObjectCandidate[] = [];
  let coveredUntil = 0;
  for (let position = 0; position < starts.length; position += 1) {
    const start = starts[position] ?? 0;
    if (start < coveredUntil) continue;
    const remainder = text.slice(start);
    const completeEnd = scanClosers(remainder).end;
    const boundary = completeEnd !== null ? start + completeEnd : starts[position + 1] ?? text.length;
    const raw = text.slice(start, boundary).trimEnd();
    const end = start + raw.length;
    const value = repairJson(raw);
    candidates.push({ start, end, raw, value });
    if (completeEnd !== null) coveredUntil = start + completeEnd;
  }
  return candidates;
}

export function looksLikeToolJson(text: string): boolean {
  return /"(?:name|tool)"\s*:/.test(text) && /"arguments"\s*:/.test(text);
}
