/** Recognizes harness tool-result blocks the model echoes verbatim. */

/*
 * Agent clients wrap tool results with their own protocol:
 *   <tool_call_result><toolcall_status>Done</toolcall_status>
 *   <command_id>job-...</command_id><command_run_logs>...</command_run_logs>
 * The model sometimes emits this whole block as its answer. It is not a tool
 * call: the block must stay hidden and the turn is then an empty recoverable
 * turn that forks to a fresh session via the existing recovery flow.
 */
export interface ResultRange { start: number; end: number }
export interface ResultTagParse { present: boolean; ranges: ResultRange[] }

interface ResultToken extends ResultRange { closing: boolean; wrapper: boolean }

const INNER = "toolcall_status|tool_call_status|process_id|terminal_id|terminal_cwd|" +
  "command_id|command_status|command_exit_code|command_exit_message|command_run_logs|skip_character_count";
const WRAPPER = "tool[ _-]?call[ _-]?result|call[ _-]?result|tool[ _-]?result";
const RESULT_TAG = new RegExp(`<\\s*(\\/?)\\s*((?:${WRAPPER}|${INNER}))\\b[^>]*>`, "gi");

export const RESULT_OPEN = new RegExp(`<\\s*(?:${WRAPPER})\\b[^>]*>`, "gi");
export const RESULT_CLOSE = new RegExp(`<\\s*\\/\\s*(?:${WRAPPER})\\b[^>]*>`, "gi");

/** Remove every result-protocol tag, leaving text. */
export function stripResultTags(text: string): string {
  RESULT_TAG.lastIndex = 0;
  return text.replace(RESULT_TAG, "");
}

function tokens(text: string): ResultToken[] {
  RESULT_TAG.lastIndex = 0;
  const list: ResultToken[] = [];
  for (const match of text.matchAll(RESULT_TAG)) {
    const start = match.index ?? 0;
    list.push({
      start, end: start + match[0].length, closing: Boolean(match[1]),
      wrapper: new RegExp(`^(?:${WRAPPER})$`, "i").test(match[2] ?? ""),
    });
  }
  return list;
}

/**
 * Extract result blocks plus the ranges that must never reach the user.
 * A wrapper covers everything until its close (or the end if truncated);
 * inner tags stranded without a wrapper form one contiguous garbage run.
 */
export function parseResultProtocol(text: string): ResultTagParse {
  const all = tokens(text);
  const wrappers = all.filter((token) => token.wrapper && !token.closing);
  const present = wrappers.length > 0 || all.filter((t) => !t.wrapper).length >= 2;
  if (!present) return { present, ranges: [] };

  const ranges: ResultRange[] = [];
  for (const opener of wrappers) {
    let close: ResultToken | undefined;
    for (const token of all) {
      if (token.start > opener.start && token.wrapper && token.closing) { close = token; break; }
    }
    ranges.push({ start: opener.start, end: close ? close.end : text.length });
  }
  // Stray inner tags outside any wrapper: hide the whole first-to-last run.
  const inside = (range: ResultRange): boolean =>
    ranges.some((block) => range.start >= block.start && range.end <= block.end);
  const stray = all.filter((token) => !token.wrapper && !inside(token));
  if (stray.length > 0) {
    ranges.push({ start: stray[0]!.start, end: stray[stray.length - 1]!.end });
  }
  return { present, ranges };
}
