/** Parses the plain-XML invoke protocol emitted with bare invoke tags. */
export interface XmlInvokeRange { start: number; end: number }
export interface XmlInvokeCandidate {
  order: number;
  consume: XmlInvokeRange;
  payload: { name: string; arguments: Record<string, unknown> };
}
export interface XmlInvokeParse { present: boolean; calls: XmlInvokeCandidate[]; ranges: XmlInvokeRange[] }

type XmlKind = "invoke" | "parameter";
interface XmlToken extends XmlInvokeRange { closing: boolean; kind: XmlKind; attributes: string }
interface XmlElement { range: XmlInvokeRange; bodyStart: number; bodyEnd: number; attributes: string }

const XML_TAG = /<\s*(\/?)\s*(invoke|parameter)\b([^>]*)>/gi;
const XML_ENTITY = /&(lt|gt|amp|quot|apos|#(\d+)|#x([0-9a-f]+));/gi;

export const XML_INVOKE_OPEN = /<\s*invoke\b[^>]*>/gi;
export const XML_INVOKE_CLOSE = /<\s*\/\s*invoke\b[^>]*>/gi;

/** Remove plain invoke/parameter tags, leaving their text. */
export function stripXmlInvokeTags(text: string): string {
  XML_TAG.lastIndex = 0;
  return text.replace(XML_TAG, "");
}

function attr(attributes: string, name: string): string {
  return attributes.match(new RegExp(`\\b${name}\\s*=\\s*["']([^"']*)["']`, "i"))?.[1]?.trim() ?? "";
}

function xmlTokens(text: string): XmlToken[] {
  XML_TAG.lastIndex = 0;
  const tokens: XmlToken[] = [];
  for (const match of text.matchAll(XML_TAG)) {
    const kind = (match[2] ?? "").toLowerCase() as XmlKind;
    const start = match.index ?? 0;
    tokens.push({
      start, end: start + match[0].length, closing: Boolean(match[1]), kind, attributes: match[3] ?? "",
    });
  }
  return tokens;
}

/** Pair open tags with the next accepted close (parameters accept invoke close). */
function xmlElements(
  text: string, tokens: readonly XmlToken[], kind: XmlKind,
  closeKinds: readonly XmlKind[] = [kind],
): XmlElement[] {
  const elements: XmlElement[] = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const open = tokens[index];
    if (!open || open.closing || open.kind !== kind) continue;
    let closeIndex = index + 1;
    let close = tokens[closeIndex];
    while (closeIndex < tokens.length && !(close?.closing && closeKinds.includes(close.kind))) {
      closeIndex += 1;
      close = tokens[closeIndex];
    }
    close = tokens[closeIndex];
    elements.push({
      range: { start: open.start, end: close ? close.end : text.length },
      bodyStart: open.end, bodyEnd: close ? close.start : text.length, attributes: open.attributes,
    });
  }
  return elements;
}

/** Name attribute of the nearest invoke opener before ``position``. */
function precedingInvokeName(tokens: readonly XmlToken[], position: number): string {
  let name = "";
  for (const token of tokens) {
    if (token.start >= position) break;
    if (!token.closing && token.kind === "invoke") name = attr(token.attributes, "name");
  }
  return name;
}

/** Decode one layer of standard XML entities and character references. */
function unescapeXml(value: string): string {
  XML_ENTITY.lastIndex = 0;
  return value.replace(XML_ENTITY, (_match, named: string, dec?: string, hex?: string) => {
    if (dec !== undefined) return String.fromCodePoint(Number(dec));
    if (hex !== undefined) return String.fromCodePoint(parseInt(hex, 16));
    return ({ lt: "<", gt: ">", amp: "&", quot: '"', apos: "'" })[named] ?? "";
  });
}

function tryJson(value: string): unknown {
  try { return JSON.parse(value.trim()) as unknown; } catch { return undefined; }
}

function parameterValue(raw: string, stringAttribute: string | undefined): unknown {
  if (stringAttribute !== undefined) {
    if (stringAttribute !== "true") {
      const json = tryJson(raw);
      if (json !== undefined) return json;
    }
    return unescapeXml(raw);
  }
  const trimmed = raw.trim();
  if (/^(?:[{[]|true\b|false\b|null\b|-?\d)/.test(trimmed)) {
    const json = tryJson(trimmed);
    if (json !== undefined) return json;
  }
  return unescapeXml(raw);
}

/** Extract bare-XML invokes plus the ranges that must never reach the user. */
export function parseXmlInvokeProtocol(text: string): XmlInvokeParse {
  const tokens = xmlTokens(text);
  const present = tokens.some((token) => token.kind === "invoke");
  if (!present) return { present, calls: [], ranges: [] };

  const invokes = xmlElements(text, tokens, "invoke");
  const parameters = xmlElements(text, tokens, "parameter", ["parameter", "invoke"]);
  const insideInvoke = (range: XmlInvokeRange): boolean =>
    invokes.some((invoke) => range.start >= invoke.range.start && range.end <= invoke.range.end);
  const ranges: XmlInvokeRange[] = [...invokes.map((invoke) => invoke.range)];
  for (const parameter of parameters) {
    if (!insideInvoke(parameter.range)) ranges.push(parameter.range);
  }

  const parameterArgs = (owner: XmlElement): Record<string, unknown> => {
    const args: Record<string, unknown> = {};
    for (const parameter of parameters) {
      if (parameter.range.start < owner.range.start || parameter.range.end > owner.range.end) continue;
      const parameterName = attr(parameter.attributes, "name");
      if (!parameterName) continue;
      args[parameterName] = parameterValue(
        text.slice(parameter.bodyStart, parameter.bodyEnd),
        attr(parameter.attributes, "string") || undefined,
      );
    }
    return args;
  };

  const calls: XmlInvokeCandidate[] = [];
  for (const invoke of invokes) {
    const name = attr(invoke.attributes, "name");
    if (name) calls.push({
      order: invoke.range.start, consume: invoke.range,
      payload: { name, arguments: parameterArgs(invoke) },
    });
  }
  // Stranded parameters (missing opener or shifted close) reuse the nearest
  // preceding invoke name; their ranges are hidden already.
  for (const parameter of parameters) {
    if (insideInvoke(parameter.range)) continue;
    const name = precedingInvokeName(tokens, parameter.range.start);
    const parameterName = attr(parameter.attributes, "name");
    if (!name || !parameterName) continue;
    const value = parameterValue(
      text.slice(parameter.bodyStart, parameter.bodyEnd),
      attr(parameter.attributes, "string") || undefined,
    );
    calls.push({
      order: parameter.range.start, consume: parameter.range,
      payload: { name, arguments: { [parameterName]: value } },
    });
  }
  return { present, calls, ranges };
}
