/** Parses the text protocol used to emulate OpenAI function calling. */
import { createHash } from "node:crypto";

import { isRecord } from "../utils/json.js";
import { collectJsonObjects, looksLikeToolJson } from "./toolCallJson.js";
import { parseDsmlProtocol, bareWrapperResidueOnly, stripBareWrapperTags, stripDsmlTags } from "./dsmlToolCalls.js";

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}
export interface ParsedToolCalls { content: string; toolCalls: OpenAIToolCall[] }
/** Registered tool shape used to recover a call whose name was omitted. */
export interface ToolDefHint { name: string; paramKeys: string[] }
export interface ParseToolHints { tools?: readonly ToolDefHint[] | undefined }

interface Range { start: number; end: number }
interface PayloadCandidate {
  order: number; consume: Range;
  payload: { name: string; arguments: Record<string, unknown> };
}
interface TagBlock extends Range { bodyStart: number; bodyEnd: number; attributes: string }

const TOOL_TAG = /<\s*(\/?)\s*(tool[_-]?call|_?call)\b([^>]*)>/gi;
// Partial DSML fragment at the end of a recovered block (close tag cut mid-way).
const PARTIAL_DSML = /<\/?[｜]{1,2}(?:DSML(?:[｜]{1,2}[ \t]*[a-z_]*)?|[DSML]{0,4})\s*$/i;
// Native DSML invokes are parsed in dsmlToolCalls.ts.

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (!isRecord(value)) return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]));
}

export function stableJson(value: unknown, pretty = false): string {
  return JSON.stringify(stableValue(value), null, pretty ? 2 : undefined) ?? "";
}

function argumentObject(value: unknown): Record<string, unknown> | null {
  if (isRecord(value)) {
    if (isRecord(value.arguments) && Object.keys(value).length === 1) {
      return argumentObject(value.arguments);
    }
    return value;
  }
  if (typeof value !== "string") return value === undefined ? {} : null;
  try {
    return argumentObject(JSON.parse(value));
  } catch {
    return null;
  }
}

/**
 * Recover an unwrapped shell parameter object {"command":"...", ...}: only
 * execution-relevant fields are kept so a strict schema ignores harness data.
 */
function shellCommandPayload(value: unknown): { name: string; arguments: Record<string, unknown> } | null {
  if (!isRecord(value) || typeof value.command !== "string" || !value.command.trim()) return null;
  const args: Record<string, unknown> = { command: value.command };
  if (typeof value.cwd === "string" && value.cwd.trim()) args.cwd = value.cwd;
  return { name: "bash", arguments: args };
}

/** Infer an omitted name: unique registered tool explaining every arg key. */
function inferToolName(args: Record<string, unknown>, tools: readonly ToolDefHint[]): string {
  const keys = Object.keys(args);
  const matches = tools.filter(
    (tool) => keys.length > 0 && keys.every((key) => tool.paramKeys.includes(key)),
  );
  return matches.length === 1 ? matches[0]?.name ?? "" : "";
}

function callPayload(
  value: unknown, attributeName = "", tools: readonly ToolDefHint[] = [],
): { name: string; arguments: Record<string, unknown> } | null {
  if (!isRecord(value)) return null;
  const nested = isRecord(value.function) ? value.function : value;
  const name =
    (typeof nested.name === "string" ? nested.name.trim() : "") ||
    (typeof value.name === "string" ? value.name.trim() : "") ||
    attributeName.trim();
  let argumentsValue = argumentObject(nested.arguments);
  if (!argumentsValue) {
    const rest = Object.fromEntries(
      Object.entries(nested).filter(([key]) => key !== "name" && key !== "function"),
    );
    argumentsValue = Object.keys(rest).length > 0 ? argumentObject(rest) : null;
  }
  if (name && argumentsValue) return { name, arguments: argumentsValue };
  if (!name && argumentsValue) {
    // Name omitted: match registered shape, else try the shell-command form.
    const inferred = inferToolName(argumentsValue, tools);
    if (inferred) return { name: inferred, arguments: argumentsValue };
  }
  return name ? null : shellCommandPayload(nested);
}

function callId(seed: string, index: number, payload: string): string {
  const digest = createHash("sha256").update(`${seed}:${index}:${payload}`).digest("hex").slice(0, 24);
  return `call_${digest}`;
}

export function formatToolCall(name: string, argumentsValue: unknown): string | null {
  const argumentsObject = argumentObject(argumentsValue);
  if (!name.trim() || !argumentsObject) return null;
  return `<tool_call>\n${stableJson({ name: name.trim(), arguments: argumentsObject })}\n</tool_call>`;
}

function attributeName(attributes: string): string {
  return attributes.match(/\bname\s*=\s*["']([^"']+)["']/i)?.[1]?.trim() ?? "";
}

function tagBlocks(text: string): TagBlock[] {
  TOOL_TAG.lastIndex = 0;
  const tags = [...text.matchAll(TOOL_TAG)].map((match) => ({
    start: match.index ?? 0, end: (match.index ?? 0) + match[0].length,
    closing: Boolean(match[1]), attributes: match[3] ?? "",
  }));
  const blocks: TagBlock[] = [];
  for (let index = 0; index < tags.length; index += 1) {
    const open = tags[index];
    if (!open || open.closing) continue;
    const next = tags[index + 1];
    if (next && !next.closing) continue;
    blocks.push({
      start: open.start, end: next?.end ?? text.length,
      bodyStart: open.end, bodyEnd: next?.start ?? text.length, attributes: open.attributes,
    });
    if (next) index += 1;
  }
  return blocks;
}

function overlaps(range: Range, blocked: readonly Range[]): boolean {
  return blocked.some((item) => range.start < item.end && range.end > item.start);
}

function removeRanges(text: string, ranges: readonly Range[]): string {
  const sorted = [...ranges].filter((range) => range.end > range.start)
    .sort((left, right) => left.start - right.start || right.end - left.end);
  let content = "";
  let cursor = 0;
  for (const range of sorted) {
    if (range.end <= cursor) continue;
    content += text.slice(cursor, Math.max(cursor, range.start));
    cursor = range.end;
  }
  return content + text.slice(cursor);
}

function withoutToolTags(text: string): string {
  TOOL_TAG.lastIndex = 0;
  return text.replace(TOOL_TAG, "");
}

function protocolOnly(text: string, ranges: readonly Range[], dsmlPresent = false): boolean {
  const stripped = withoutToolTags(removeRanges(text, ranges));
  return (dsmlPresent ? stripDsmlTags(stripped) : stripped).trim().length === 0;
}

function pushUnique(
  toolCalls: OpenAIToolCall[], seed: string,
  payload: { name: string; arguments: Record<string, unknown> }, seen: Set<string>,
): void {
  const argumentsText = stableJson(payload.arguments);
  const key = `${payload.name}\0${argumentsText}`;
  if (seen.has(key)) return;
  seen.add(key);
  toolCalls.push({
    id: callId(seed, toolCalls.length, `${payload.name}:${argumentsText}`),
    type: "function", function: { name: payload.name, arguments: argumentsText },
  });
}

function taggedCandidates(
  text: string, blocks: readonly TagBlock[], tools: readonly ToolDefHint[],
): { calls: PayloadCandidate[]; artifacts: Range[] } {
  const calls: PayloadCandidate[] = [];
  const artifacts: Range[] = [];
  for (const block of blocks) {
    const body = text.slice(block.bodyStart, block.bodyEnd);
    const name = attributeName(block.attributes);
    // First standalone object is always a candidate inside a tool-call tag.
    const objects = collectJsonObjects(body, true);
    let valid = false;
    for (const object of objects) {
      const payload = callPayload(object.value, name, tools);
      if (!payload) continue;
      valid = true;
      calls.push({ order: block.bodyStart + object.start, consume: block, payload });
    }
    if (!valid && (looksLikeToolJson(body) || (name && /"arguments"\s*:/.test(body)))) {
      artifacts.push(block);
    }
  }
  return { calls, artifacts };
}

/** Parse model tool-call text into OpenAI-compatible calls and cleaned content. */
export function parseToolCalls(
  text: string, seed = "tool", hints: ParseToolHints = {},
): ParsedToolCalls {
  const tools = hints.tools ?? [];
  const blocks = tagBlocks(text);
  const tagged = taggedCandidates(text, blocks, tools);
  const dsml = parseDsmlProtocol(text);
  const blocked: readonly Range[] = [...blocks, ...dsml.ranges];
  const bareObjects = collectJsonObjects(text).filter(
    (object) => !overlaps({ start: object.start, end: object.end }, blocked),
  );
  const bareRanges = bareObjects
    .filter((object) => looksLikeToolJson(object.raw))
    .map((object) => ({ start: object.start, end: object.end }));
  const sweepRanges = [...blocks, ...dsml.ranges, ...bareRanges];
  const bareContext = protocolOnly(text, sweepRanges, dsml.present);
  const bareCalls: PayloadCandidate[] = bareContext
    ? bareObjects.flatMap((object) => {
        const payload = callPayload(object.value, "", tools);
        return payload
          ? [{ order: object.start, consume: { start: object.start, end: object.end }, payload }]
          : [];
      })
    : [];
  const candidates = [...tagged.calls, ...dsml.calls, ...bareCalls].sort(
    (left, right) => left.order - right.order,
  );
  const toolCalls: OpenAIToolCall[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) pushUnique(toolCalls, seed, candidate.payload, seen);

  const artifactRanges = [...tagged.artifacts, ...bareRanges];
  const residueOnly = !dsml.present && bareWrapperResidueOnly(text);
  const stripProtocolTags = dsml.present || residueOnly;
  const isProtocolOnly =
    (artifactRanges.length > 0 || stripProtocolTags) &&
    protocolOnly(text, sweepRanges, stripProtocolTags);
  if (toolCalls.length === 0 && !isProtocolOnly && !stripProtocolTags) {
    return { content: text, toolCalls };
  }
  const consumed = candidates.map((candidate) => candidate.consume);
  const withoutArtifacts = removeRanges(text, [
    ...consumed,
    ...(toolCalls.length > 0 ? blocks : []),
    ...(bareContext || isProtocolOnly ? artifactRanges : []),
    ...(dsml.present ? dsml.ranges : []),
  ]);
  // Strip DSML tags when the protocol was present or a call was recovered,
  // even if the wrapper closer survived alone as an orphan tag.
  const withoutMarkup = withoutToolTags(
    dsml.present || toolCalls.length > 0 ? stripDsmlTags(withoutArtifacts) : withoutArtifacts,
  );
  // Bare wrapper residue can prefix ordinary <tool_call> blocks too.
  const withoutResidue =
    toolCalls.length > 0 || residueOnly ? stripBareWrapperTags(withoutMarkup) : withoutMarkup;
  const withoutPartial = toolCalls.length > 0 ? withoutResidue.replace(PARTIAL_DSML, "") : withoutResidue;
  const cleaned = withoutPartial.replace(/\n{3,}/g, "\n\n").trim();
  return { content: cleaned, toolCalls };
}

/** Prefer valid output calls, then promote calls leaked into reasoning. */
export function parseToolCallsFromParts(
  outputText: string, reasoningText: string, seed = "tool", hints: ParseToolHints = {},
): ParsedToolCalls {
  const fromOutput = parseToolCalls(outputText, seed, hints);
  if (fromOutput.toolCalls.length > 0) return fromOutput;
  const fromReasoning = parseToolCalls(reasoningText, seed, hints);
  if (fromReasoning.toolCalls.length > 0) {
    return { content: fromOutput.content, toolCalls: fromReasoning.toolCalls };
  }
  return outputText.trim() ? fromOutput : fromReasoning;
}

export function canonicalParsedAssistantText(parsed: ParsedToolCalls): string {
  const calls = parsed.toolCalls
    .map((call) => formatToolCall(call.function.name, call.function.arguments))
    .filter((call): call is string => call !== null);
  return [parsed.content, ...calls].filter(Boolean).join("\n");
}

/** Store a canonical assistant turn so structured client history can match it. */
export function canonicalAssistantText(text: string): string {
  const parsed = parseToolCalls(text);
  return parsed.toolCalls.length > 0 ? canonicalParsedAssistantText(parsed) : parsed.content;
}
