/** Disambiguates JSON control escapes that collide with Windows path text. */

/*
 * A model emitting bare Windows paths writes "d:\tmp\tijian" inside JSON.
 * Strict JSON.parse succeeds: \t is a legal tab escape, silently corrupting
 * the path, so the lenient-backslash repair (which only runs after failure)
 * never triggers. When a string value itself contains a drive prefix, a lone
 * backslash before t/b/f/n/r inside that string is treated as a literal path
 * separator: the escape is doubled before parsing.
 */
type Span = [number, number];

/** Raw string-literal spans (excluding quotes) using backslash-aware scanning. */
function stringSpans(text: string): Span[] {
  const spans: Span[] = [];
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== '"') continue;
    let end = index + 1;
    while (end < text.length && text[end] !== '"') {
      if (text[end] === "\\") end += 1;
      end += 1;
    }
    const raw = text.slice(index + 1, end);
    if (/[A-Za-z]:[\\/]/.test(raw)) spans.push([index + 1, end]);
    index = end;
  }
  return spans;
}

/** Double the final backslash of an odd run when it escapes t/b/f/n/r. */
function fixValue(raw: string): string {
  let result = "";
  for (let index = 0; index < raw.length;) {
    if (raw[index] !== "\\") { result += raw[index]; index += 1; continue; }
    let run = 0;
    while (raw[index + run] === "\\") run += 1;
    const after = raw[index + run] ?? "";
    const literal = run % 2 === 1 && /[tbnfr]/.test(after);
    result += "\\".repeat(literal ? run + 1 : run) + after;
    index += run + 1;
  }
  return result;
}

/** Rewrite ambiguous control escapes only inside drive-prefixed strings. */
export function disambiguatePathEscapes(text: string): string {
  if (!/[A-Za-z]:[\\/]/.test(text)) return text;
  const spans = stringSpans(text);
  if (spans.length === 0) return text;
  let result = "", cursor = 0;
  for (const [start, end] of spans) {
    result += text.slice(cursor, start) + fixValue(text.slice(start, end));
    cursor = end;
  }
  return result + text.slice(cursor);
}
