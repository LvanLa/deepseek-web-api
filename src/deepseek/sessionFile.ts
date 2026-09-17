/** Debounced atomic persistence for the session index with a single-instance lock. */
import fs from "node:fs";
import path from "node:path";

import type { Logger } from "../utils/logger.js";

const FLUSH_DELAY_MS = 250;

interface LockBody {
  pid: number;
  startedAt: string;
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM means the process exists but cannot be signalled; ESRCH means it is gone.
    return (error as NodeJS.ErrnoException)?.code === "EPERM";
  }
}

/** Buffers rapid turn updates into one atomic file replacement per quiet window. */
export class SessionFile {
  private timer: NodeJS.Timeout | null = null;
  private pending: string | null = null;
  private readonly lockFile: string;

  constructor(
    private readonly file: string | undefined,
    private readonly logger?: Logger,
  ) {
    this.lockFile = file ? `${file}.lock` : "";
    if (file) {
      this.acquireLock();
      process.once("exit", () => this.flushSync());
    }
  }

  read(): string | null {
    if (!this.file || !fs.existsSync(this.file)) return null;
    return fs.readFileSync(this.file, "utf8");
  }

  requestSave(data: string): void {
    if (!this.file) return;
    this.pending = data;
    if (this.timer) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.flushSync();
    }, FLUSH_DELAY_MS);
    this.timer.unref?.();
  }

  /** Write the latest buffered snapshot immediately; safe to call on shutdown. */
  flushSync(): void {
    if (!this.file || this.pending === null) return;
    const data = this.pending;
    this.pending = null;
    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true, mode: 0o700 });
      const temporary = `${this.file}.${process.pid}.tmp`;
      fs.writeFileSync(temporary, data, { encoding: "utf8", mode: 0o600 });
      fs.renameSync(temporary, this.file);
    } catch (error) {
      this.pending = data;
      this.logger?.error("failed to save sessions file", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  close(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.flushSync();
    this.releaseLock();
  }

  private acquireLock(): void {
    try {
      if (fs.existsSync(this.lockFile)) {
        const body = JSON.parse(fs.readFileSync(this.lockFile, "utf8")) as Partial<LockBody>;
        if (
          typeof body.pid === "number" &&
          body.pid !== process.pid &&
          pidAlive(body.pid)
        ) {
          this.logger?.warn("another process owns the sessions file; concurrent writers may lose turns", {
            pid: body.pid,
            file: this.file,
          });
        }
      }
      const body: LockBody = { pid: process.pid, startedAt: new Date().toISOString() };
      fs.writeFileSync(this.lockFile, JSON.stringify(body), { encoding: "utf8", mode: 0o600 });
    } catch (error) {
      this.logger?.warn("could not create sessions lock file", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private releaseLock(): void {
    if (!this.lockFile) return;
    try {
      const body = JSON.parse(fs.readFileSync(this.lockFile, "utf8")) as Partial<LockBody>;
      if (body.pid === process.pid) fs.unlinkSync(this.lockFile);
    } catch {
      // A missing or foreign lock is not an error during shutdown.
    }
  }
}
