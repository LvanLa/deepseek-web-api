/** Dependency contract injected into the HTTP routing layer. */
import type { DeepSeekClient } from "../deepseek/client.js";
import type { Logger } from "../utils/logger.js";

export interface ServerDependencies {
  client: DeepSeekClient;
  apiKeys: readonly string[];
  debug: boolean;
  logger?: Logger;
}
