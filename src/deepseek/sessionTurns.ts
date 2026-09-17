/** Matches normalized conversation windows across full and truncated client histories. */
import { createHash } from "node:crypto";

import { normalizeText } from "../utils/text.js";
import type { MessageTurn } from "./types.js";

export function fingerprint(turns: MessageTurn[]): string {
  return turns.map((turn) => `${turn.role}:${turn.content}`).join("\n---\n");
}

/** Index keys are hashed so full conversation text never accumulates on disk. */
export function fpKey(value: string): string {
  return `fp:${createHash("sha256").update(value).digest("hex")}`;
}

export function turnsEqual(left: MessageTurn[], right: MessageTurn[]): boolean {
  if (left.length === 0 || left.length !== right.length) return false;
  return left.every((turn, index) =>
    turn.role === right[index]?.role && normalizeText(turn.content) === normalizeText(right[index]?.content),
  );
}

export function turnsPrefix(left: MessageTurn[], right: MessageTurn[]): boolean {
  if (left.length === 0 || right.length === 0) return false;
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    if (left[index]?.role !== right[index]?.role) return false;
    if (normalizeText(left[index]?.content) !== normalizeText(right[index]?.content)) return false;
  }
  return true;
}

export function turnsSuffix(shorter: MessageTurn[], longer: MessageTurn[]): boolean {
  if (shorter.length === 0 || shorter.length > longer.length) return false;
  const offset = longer.length - shorter.length;
  return shorter.every((turn, index) => {
    const candidate = longer[offset + index];
    return turn.role === candidate?.role && normalizeText(turn.content) === normalizeText(candidate.content);
  });
}

/**
 * Fold expanded tool turns into canonical user/assistant rounds for matching:
 * tool/function results are dropped and adjacent assistant turns are merged,
 * matching how a completed logical assistant round is stored canonically.
 */
export function foldTurns(turns: readonly MessageTurn[]): MessageTurn[] {
  const folded: MessageTurn[] = [];
  for (const turn of turns) {
    if (turn.role === "tool" || turn.role === "function") continue;
    const last = folded.at(-1);
    if (turn.role === "assistant" && last?.role === "assistant") {
      last.content = [last.content, turn.content].filter((text) => text.trim()).join("\n");
    } else {
      folded.push({ role: turn.role, content: turn.content });
    }
  }
  return folded;
}
