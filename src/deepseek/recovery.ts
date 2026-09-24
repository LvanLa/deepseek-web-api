/** Helpers for fresh-session recovery after empty or context-length failures. */
import type { Logger } from "../utils/logger.js";
import { isContextLengthExceeded } from "./updates.js";

export interface FrameTracker { frames: number; noteFrame: () => void }

/** Counts frames already delivered to the client (unretryable once nonzero). */
export function frameTracker(): FrameTracker {
  const tracker: FrameTracker = { frames: 0, noteFrame: () => {} };
  tracker.noteFrame = () => { tracker.frames += 1; };
  return tracker;
}

/** The error when it is a context-length failure and no frame was streamed. */
export function catchContextLength(error: unknown, frames: number): unknown | null {
  return frames === 0 && isContextLengthExceeded(error) ? error : null;
}

const AUTH_TOKEN_MARKER = "deepseek-auth-token-invalid";

/** A 40003-style rejection raised by prepare: login must be refreshed. */
export function isAuthTokenError(error: unknown): boolean {
  return error instanceof Error && error.message.includes(AUTH_TOKEN_MARKER);
}

/** Log the fork decision once before rebuilding the compact retry request. */
export function logContextFork(logger: Logger, fields: Record<string, unknown>, fresh: boolean): void {
  logger.info(
    fresh
      ? "Retrying in a fresh DeepSeek session after context length limit"
      : "Retrying with a compact prompt after context length limit",
    fields,
  );
}
