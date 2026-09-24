/** Parses the native DeepSeek DSML tool protocol. */
import { isRecord } from "../utils/json.js";
import { collectJsonObjects } from "./toolCallJson.js";

/*
 * The model sometimes ignores the prompted <tool_call> JSON shape and falls
 * back to its built-in web tools (for example run_code), emitting:
 *
 *   <[bars]DSML[bars]calls>
 *     <[bars]DSML[bars]invoke name="fn">
 *       <[bars]DSML[bars]parameter name="k" string="true">raw text</...parameter>
 *       <[bars]DSML[bars]parameter name="n" string="false">42</...parameter>
 *     </...invoke>
 *   </...calls>
 *
 * "[bars]" is one or two U+FF5C full-width vertical bars per side: web streams
 * use two, open-weight models one, and a space may precede the keyword. A
 * string="true" body is verbatim text (closing tags are HTML-escaped inside
 * it); any other body is JSON text.
 */

export interface DsmlRange {
  start: number;
  end: number;
}

export interface DsmlCandidate {
  order: number;
  consume: DsmlRange;
  payload: { name: string; arguments: Record<string, unknown> };
}

export interface DsmlParse {
  present: boolean;
  calls: DsmlCandidate[];
  ranges: DsmlRange[];
}

type DsmlKind = "wrapper" | "invoke" | "parameter";

interface DsmlToken extends DsmlRange {
  closing: boolean;
  kind: DsmlKind;
  attributes: string;
}

interface DsmlElement {
  range: DsmlRange;
  bodyStart: number;
  bodyEnd: number;
  attributes: string;
}

// \s (not [ \t]) so NBSP/ideographic space between bars and the keyword do
// not make the whole DSML block fall through as assistant content.
const DSML_TAG = /<(\/?)[｜]{1,2}DSML[｜]{1,2}\s*([a-z_]+)\b([^>]*)>/gi;
const DSML_ESCAPED_CLOSE =
  /&((?:amp;)*)lt;(\/[｜]{1,2}DSML[｜]{1,2}\s*parameter)>/gi;
// Renderers sometimes mangle the wrapper opener into plain ASCII ("< calls>")
// while the rest stays DSML, leaving a residue tag with no full-width bars.
const BARE_WRAPPER_TAG = /<\/?\s*(?:calls|tool_calls|function_calls)\s*>/gi;
const DSML_WRAPPER_NAMES = new Set(["calls", "tool_calls", "function_calls"]);

// Stream-level detection shared with StreamSieve. \s (not [ \t]) tolerates
// NBSP/ideographic space before the keyword.
export const DSML_OPEN =
  /<[｜]{1,2}DSML[｜]{1,2}\s*(invoke|calls|tool_calls|function_calls)\b[^>]*>/gi;
export const DSML_CLOSE =
  /<\/[｜]{1,2}DSML[｜]{1,2}\s*(invoke|calls|tool_calls|function_calls)\b[^>]*>/gi;
export const DSML_INVOKE_OPEN = /<[｜]{1,2}DSML[｜]{1,2}\s*invoke\b[^>]*>/i;
export const DSML_INVOKE_CLOSE = /<\/[｜]{1,2}DSML[｜]{1,2}\s*invoke\b[^>]*>/i;
export const DSML_WRAPPER_CLOSE =
  /<\/[｜]{1,2}DSML[｜]{1,2}\s*(calls|tool_calls|function_calls)\b[^>]*>/i;
export const DSML_WRAPPER_NAME_LIST = ["calls", "tool_calls", "function_calls"] as const;

/** DSML opener prefix (close angle not required yet) for a wrapper name. */
export function dsmlPrefix(name: string): RegExp {
  return new RegExp(`<[｜]{1,2}DSML[｜]{1,2}\\s*${name}\\b`, "i");
}

/** Matching wrapper close in DSML form or ASCII-mangled form. */
export function dsmlBlockClose(name: string): RegExp {
  return new RegExp(
    `(?:</[｜]{1,2}DSML[｜]{1,2}\\s*${name}\\b[^>]*>|<\\s*\\/\\s*${name}\\s*>)`, "i");
}

/** Remove every DSML tag and any ASCII wrapper residue, leaving plain text. */
export function stripDsmlTags(text: string): string {
  DSML_TAG.lastIndex = 0;
  return stripBareWrapperTags(text.replace(DSML_TAG, ""));
}

/** Remove ASCII-only wrapper leftovers such as "< calls>" or "</tool_calls>". */
export function stripBareWrapperTags(text: string): string {
  BARE_WRAPPER_TAG.lastIndex = 0;
  return text.replace(BARE_WRAPPER_TAG, "");
}

/** True when the text contains only wrapper leftovers and whitespace. */
export function bareWrapperResidueOnly(text: string): boolean {
  BARE_WRAPPER_TAG.lastIndex = 0;
  return BARE_WRAPPER_TAG.test(text) && stripBareWrapperTags(text).trim().length === 0;
}

function attributeValue(attributes: string, name: string): string {
  return (
    attributes
      .match(new RegExp(`\\b${name}\\s*=\\s*["']([^"']*)["']`, "i"))?.[1]
      ?.trim() ?? ""
  );
}

function dsmlKind(keyword: string): DsmlKind | null {
  if (keyword === "invoke") return "invoke";
  if (keyword === "parameter") return "parameter";
  return DSML_WRAPPER_NAMES.has(keyword) ? "wrapper" : null;
}

function dsmlTokens(text: string): DsmlToken[] {
  DSML_TAG.lastIndex = 0;
  const tokens: DsmlToken[] = [];
  for (const match of text.matchAll(DSML_TAG)) {
    const kind = dsmlKind((match[2] ?? "").toLowerCase());
    if (!kind) continue;
    const start = match.index ?? 0;
    tokens.push({
      start,
      end: start + match[0].length,
      closing: Boolean(match[1]),
      kind,
      attributes: match[3] ?? "",
    });
  }
  return tokens;
}

/**
 * Pair each open tag with the next close of an accepted kind (loose nesting).
 * Parameters accept an invoke close too, because malformed streams frequently
 * close a parameter with a mismatched "</...invoke>" tag.
 */
function dsmlElements(
  text: string,
  tokens: readonly DsmlToken[],
  kind: DsmlKind,
  closeKinds: readonly DsmlKind[] = [kind],
): DsmlElement[] {
  const elements: DsmlElement[] = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const open = tokens[index];
    if (!open || open.closing || open.kind !== kind) continue;
    let closeIndex = index + 1;
    let closeToken = tokens[closeIndex];
    while (closeIndex < tokens.length &&
      !(closeToken?.closing && closeKinds.includes(closeToken.kind))) {
      closeIndex += 1;
      closeToken = tokens[closeIndex];
    }
    const close = tokens[closeIndex];
    elements.push({
      range: { start: open.start, end: close ? close.end : text.length },
      bodyStart: open.end,
      bodyEnd: close ? close.start : text.length,
      attributes: open.attributes,
    });
  }
  return elements;
}

/** Name attribute of the nearest invoke opener starting before ``position``. */
function precedingInvokeName(tokens: readonly DsmlToken[], position: number): string {
  let name = "";
  for (const token of tokens) {
    if (token.start >= position) break;
    if (!token.closing && token.kind === "invoke") name = attributeValue(token.attributes, "name");
  }
  return name;
}

/** Restore one layer of the closing-tag escaping used inside DSML string bodies. */
function unescapeDsmlValue(value: string): string {
  DSML_ESCAPED_CLOSE.lastIndex = 0;
  return value.replace(
    DSML_ESCAPED_CLOSE,
    (_match, amps: string, tail: string) =>
      amps ? `&${amps.slice(4)}lt;${tail}>` : `<${tail}>`,
  );
}

function parseDsmlJson(value: string): unknown {
  try {
    return JSON.parse(value.trim()) as unknown;
  } catch {
    return undefined;
  }
}

function dsmlParameterValue(raw: string, stringAttribute: string | undefined): unknown {
  if (stringAttribute !== undefined) {
    if (stringAttribute !== "true") {
      const json = parseDsmlJson(raw);
      if (json !== undefined) return json;
    }
    return unescapeDsmlValue(raw);
  }
  // No type attribute: trust JSON-shaped primitives/structures, keep the rest raw.
  const trimmed = raw.trim();
  if (/^(?:[{[]|true\b|false\b|null\b|-?\d)/.test(trimmed)) {
    const json = parseDsmlJson(trimmed);
    if (json !== undefined) return json;
  }
  return unescapeDsmlValue(raw);
}

/** Extract native DSML invokes plus the ranges that must never reach the user. */
export function parseDsmlProtocol(text: string): DsmlParse {
  const tokens = dsmlTokens(text);
  const wrappers = dsmlElements(text, tokens, "wrapper");
  // A wrapper opener alone (stream truncated, no invoke tags) still counts.
  const present = tokens.some((token) => token.kind === "invoke") || wrappers.length > 0;
  if (!present) return { present, calls: [], ranges: [] };

  const invokes = dsmlElements(text, tokens, "invoke");
  // Accept invoke closes as parameter bounds for malformed streams.
  const parameters = dsmlElements(text, tokens, "parameter", ["parameter", "invoke"]);
  const isInsideInvoke = (range: DsmlRange): boolean =>
    invokes.some(
      (invoke) => range.start >= invoke.range.start && range.end <= invoke.range.end,
    );
  const ranges: DsmlRange[] = [
    ...wrappers.map((element) => element.range),
    ...invokes.map((element) => element.range),
  ];
  // Parameters outside any invoke are protocol garbage, not user text.
  for (const parameter of parameters) {
    if (!isInsideInvoke(parameter.range)) ranges.push(parameter.range);
  }
  // Mangled ASCII wrapper leftovers (for example "< calls>") must be swept too.
  BARE_WRAPPER_TAG.lastIndex = 0;
  for (const match of text.matchAll(BARE_WRAPPER_TAG)) {
    const range = { start: match.index ?? 0, end: (match.index ?? 0) + match[0].length };
    if (!isInsideInvoke(range)) ranges.push(range);
  }

  const parameterArgs = (owner: DsmlElement): Record<string, unknown> => {
    const args: Record<string, unknown> = {};
    for (const parameter of parameters) {
      if (parameter.range.start < owner.range.start || parameter.range.end > owner.range.end) continue;
      const parameterName = attributeValue(parameter.attributes, "name");
      if (!parameterName) continue;
      args[parameterName] = dsmlParameterValue(
        text.slice(parameter.bodyStart, parameter.bodyEnd),
        attributeValue(parameter.attributes, "string") || undefined,
      );
    }
    return args;
  };

  const calls: DsmlCandidate[] = [];
  for (const invoke of invokes) {
    const name = attributeValue(invoke.attributes, "name");
    if (name) calls.push({
      order: invoke.range.start,
      consume: invoke.range,
      payload: { name, arguments: parameterArgs(invoke) },
    });
  }
  // Recover parameters stranded outside any invoke (missing opener or shifted
  // close tags): rebuild one call per parameter, reusing the nearest preceding
  // invoke name. Their ranges are already hidden, so nothing extra leaks.
  for (const parameter of parameters) {
    if (isInsideInvoke(parameter.range)) continue;
    const name = precedingInvokeName(tokens, parameter.range.start);
    const parameterName = attributeValue(parameter.attributes, "name");
    if (!name || !parameterName) continue;
    const value = dsmlParameterValue(text.slice(parameter.bodyStart, parameter.bodyEnd),
      attributeValue(parameter.attributes, "string") || undefined);
    calls.push({ order: parameter.range.start, consume: parameter.range,
      payload: { name, arguments: { [parameterName]: value } } });
  }
  // Malformed wrapper whose body is bare JSON (no invoke opener, shifted
  // close tags): recover tool-shaped objects from the wrapper body directly.
  for (const wrapper of wrappers) {
    // Only malformed wrappers (no invoke opener inside) get bare recovery.
    const hasInvoke = invokes.some((invoke) =>
      invoke.range.start >= wrapper.range.start && invoke.range.end <= wrapper.range.end);
    if (hasInvoke) continue;
    const body = text.slice(wrapper.bodyStart, wrapper.bodyEnd);
    for (const object of collectJsonObjects(body, true)) {
      if (!isRecord(object.value)) continue;
      const rawName = object.value.name ?? object.value.tool;
      const name = typeof rawName === "string" ? rawName.trim() : "";
      const args = isRecord(object.value.arguments) ? object.value.arguments : null;
      if (!name || !args) continue;
      calls.push({ order: wrapper.range.start + object.start, consume: wrapper.range,
        payload: { name, arguments: args } });
    }
  }
  return { present, calls, ranges };
}
