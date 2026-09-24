/** Parses the text protocol used to emulate OpenAI function calling. */
import { createHash } from "node:crypto";

import { isRecord } from "../utils/json.js";
import { collectJsonObjects, looksLikeToolJson } from "./toolCallJson.js";
import { parseDsmlProtocol, bareWrapperResidueOnly, stripBareWrapperTags, stripDsmlTags } from "./dsmlToolCalls.js";
import { parseXmlInvokeProtocol, stripXmlInvokeTags } from "./xmlToolCalls.js";
import { parseResultProtocol, stripResultTags } from "./toolResultTags.js";

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
// Degenerate opener "<>" left when the tag name drifts into the THINK channel.
const EMPTY_ANGLE = /<\s*>/g;
// Partial DSML fragment at the end of a recovered block (close tag cut mid-way).
const PARTIAL_DSML = /<\/?[｜]{1,2}(?:DSML(?:[｜]{1,2}\s*[a-z_]*)?|[DSML]{0,4})\s*$/i;
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
    return isRecord(value.arguments) && Object.keys(value).length === 1
      ? argumentObject(value.arguments)
      : value;
  }
  if (typeof value !== "string") return value === undefined ? {} : null;
  try { return argumentObject(JSON.parse(value)); } catch { return null; }
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
    (typeof nested.tool === "string" ? nested.tool.trim() : "") ||
    (typeof value.name === "string" ? value.name.trim() : "") ||
    attributeName.trim();
  let argumentsValue = argumentObject(nested.arguments);
  if (!argumentsValue) {
    const rest = Object.fromEntries(
      Object.entries(nested).filter(([key]) => key !== "name" && key !== "tool" && key !== "function"),
    );
    argumentsValue = Object.keys(rest).length ? argumentObject(rest) : null;
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
  const tags = [...text.matchAll(TOOL_TAG)].map((match) => {
    const start = match.index ?? 0;
    return { start, end: start + match[0].length, closing: Boolean(match[1]), attributes: match[3] ?? "" };
  });
  const blocks: TagBlock[] = [];
  for (let index = 0; index < tags.length; index += 1) {
    const open = tags[index]!;
    if (open.closing) continue;
    const next = tags[index + 1];
    if (next && !next.closing) continue;
    blocks.push({ start: open.start, end: next?.end ?? text.length,
      bodyStart: open.end, bodyEnd: next?.start ?? text.length, attributes: open.attributes });
    if (next) index += 1;
  }
  return blocks;
}

function overlaps(range: Range, blocked: readonly Range[]): boolean {
  return blocked.some((item) => range.start < item.end && range.end > item.start);
}

function matchRanges(text: string, pattern: RegExp): Range[] {
  pattern.lastIndex = 0;
  return [...text.matchAll(pattern)].map((match) =>
    ({ start: match.index ?? 0, end: (match.index ?? 0) + match[0].length }));
}

function removeRanges(text: string, ranges: readonly Range[]): string {
  const sorted = [...ranges].filter((range) => range.end > range.start)
    .sort((left, right) => left.start - right.start || right.end - left.end);
  let content = "", cursor = 0;
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

function protocolOnly(text: string, ranges: readonly Range[], dsmlPresent = false,
  bareSeparators = false, xmlPresent = false, resultPresent = false): boolean {
  let residue = withoutToolTags(removeRanges(text, ranges));
  if (dsmlPresent) residue = stripDsmlTags(residue);
  if (xmlPresent) residue = stripXmlInvokeTags(residue);
  if (resultPresent) residue = stripResultTags(residue);
  if (bareSeparators) residue = residue.replace(/[\s,]+/g, "");
  return residue.length === 0;
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
  const tools = hints.tools ?? []; const blocks = tagBlocks(text);
  const tagged = taggedCandidates(text, blocks, tools);
  const dsml = parseDsmlProtocol(text);
  const xml = parseXmlInvokeProtocol(text);
  const result = parseResultProtocol(text);
  const blocked: readonly Range[] = [...blocks, ...dsml.ranges, ...xml.ranges, ...result.ranges];
  const bareObjects = collectJsonObjects(text).filter(
    (object) => !overlaps({ start: object.start, end: object.end }, blocked),
  );
  const bareRanges = bareObjects
    .filter((object) => looksLikeToolJson(object.raw))
    .map((object) => ({ start: object.start, end: object.end }));
  // Empty angle pairs count as residue only when a tool-shaped JSON is present.
  const emptyRanges = bareRanges.length > 0 ? matchRanges(text, EMPTY_ANGLE) : [];
  const sweepRanges = [...blocks, ...dsml.ranges, ...xml.ranges, ...result.ranges, ...bareRanges, ...emptyRanges];
  const bareContext = protocolOnly(text, sweepRanges, dsml.present, true, xml.present, result.present);
  // Gaps between bare objects are their separators; remove them with the calls.
  const bareGapRanges: Range[] = bareContext
    ? bareRanges.slice(1).map((range, index) => ({ start: bareRanges[index]!.end, end: range.start })) : [];
  const bareCalls: PayloadCandidate[] = bareContext
    ? bareObjects.flatMap((object) => {
        const payload = callPayload(object.value, "", tools);
        return payload ? [{ order: object.start, consume: { start: object.start, end: object.end }, payload }] : [];
      })
    : [];
  const candidates = [...tagged.calls, ...dsml.calls, ...xml.calls, ...bareCalls]
    .sort((left, right) => left.order - right.order);
  const toolCalls: OpenAIToolCall[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) pushUnique(toolCalls, seed, candidate.payload, seen);

  const artifactRanges = [...tagged.artifacts, ...bareRanges];
  const residueOnly = !dsml.present && bareWrapperResidueOnly(text);
  const stripProtocolTags = dsml.present || xml.present || result.present || residueOnly;
  const isProtocolOnly =
    (artifactRanges.length > 0 || stripProtocolTags) &&
    protocolOnly(text, sweepRanges, stripProtocolTags, false, xml.present, result.present);
  if (toolCalls.length === 0 && !isProtocolOnly && !stripProtocolTags) return { content: text, toolCalls };

  const consumed = candidates.map((candidate) => candidate.consume);
  const withoutArtifacts = removeRanges(text, [
    ...consumed, ...(toolCalls.length > 0 ? blocks : []),
    ...(bareContext || isProtocolOnly ? artifactRanges : []),
    ...(bareContext ? emptyRanges : []), ...(bareContext ? bareGapRanges : []),
    ...(dsml.present ? dsml.ranges : []), ...(xml.present ? xml.ranges : []),
    ...(result.present ? result.ranges : []),
  ]);
  // Strip protocol tags when the family was present or a call was recovered,
  // even if the wrapper closer survived alone as an orphan tag.
  const needsStrip = dsml.present || xml.present || result.present || toolCalls.length > 0;
  const strippedTags = needsStrip
    ? stripResultTags(stripXmlInvokeTags(stripDsmlTags(withoutArtifacts)))
    : withoutArtifacts;
  const withoutMarkup = withoutToolTags(strippedTags);
  // Bare wrapper residue can prefix ordinary <tool_call> blocks too.
  const withoutResidue =
    toolCalls.length > 0 || residueOnly ? stripBareWrapperTags(withoutMarkup) : withoutMarkup;
  const withoutPartial = toolCalls.length > 0 ? withoutResidue.replace(PARTIAL_DSML, "") : withoutResidue;
  const withoutEmpty = toolCalls.length > 0 ? withoutPartial.replace(/^\s*<\s*>/, "") : withoutPartial;
  const cleaned = withoutEmpty.replace(/\n{3,}/g, "\n\n").trim();
  return { content: cleaned, toolCalls };
}

/** Prefer valid output calls, then promote calls leaked into reasoning. */
export function parseToolCallsFromParts(
  outputText: string, reasoningText: string, seed = "tool", hints: ParseToolHints = {},
): ParsedToolCalls {
  const fromOutput = parseToolCalls(outputText, seed, hints);
  if (fromOutput.toolCalls.length > 0) return fromOutput;
  const fromReasoning = parseToolCalls(reasoningText, seed, hints);
  if (fromReasoning.toolCalls.length > 0) { return { content: fromOutput.content, toolCalls: fromReasoning.toolCalls }; }
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
