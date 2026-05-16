const { chromium } = require("playwright");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const child_process = require("child_process");
const os = require("os");

const CDP_URL = process.env.GPT_WEB_CDP_URL || "http://127.0.0.1:9222";

// ---------------------------------------------------------------------------
// Logging.  Everything in this driver routes through `vlog` so every line
// carries a wallclock timestamp and an elapsed-since-start counter.  When a
// chat invocation looks frozen, the log is the only ground truth.
// In addition to stdout (captured by the Python orchestrator into chat.log),
// every log line is appended to debug.log inside the run's metadata dir.
// ---------------------------------------------------------------------------
const T0 = Date.now();
let DEBUG_LOG_PATH = null;

function tsPrefix() {
  const iso = new Date().toISOString();
  const elapsed = ((Date.now() - T0) / 1000).toFixed(1).padStart(7, " ");
  return `[${iso}] [t+${elapsed}s]`;
}

function formatLogPart(part) {
  if (typeof part === "string") return part;
  try {
    return JSON.stringify(part);
  } catch (_) {
    return String(part);
  }
}

function vlog(...parts) {
  const line = `${tsPrefix()} ${parts.map(formatLogPart).join(" ")}`;
  // eslint-disable-next-line no-console
  console.log(line);
  if (DEBUG_LOG_PATH) {
    try {
      fs.appendFileSync(DEBUG_LOG_PATH, line + "\n", "utf8");
    } catch (_) {}
  }
}

function vwarn(...parts) {
  const line = `${tsPrefix()} WARN ${parts.map(formatLogPart).join(" ")}`;
  // eslint-disable-next-line no-console
  console.log(line);
  if (DEBUG_LOG_PATH) {
    try {
      fs.appendFileSync(DEBUG_LOG_PATH, line + "\n", "utf8");
    } catch (_) {}
  }
}

function setDebugLog(p) {
  DEBUG_LOG_PATH = p;
  try {
    mkdirp(path.dirname(p));
    fs.writeFileSync(p, "", "utf8");
  } catch (err) {
    vwarn("could not initialize debug log", { path: p, err: err.message });
  }
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const key = argv[i];
    if (!key.startsWith("--")) continue;
    const name = key.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[name] = "true";
    } else {
      args[name] = next;
      i++;
    }
  }
  return args;
}

function boolArg(args, name) {
  return /^(1|true|yes|y)$/i.test(String(args[name] || ""));
}

function intArg(args, name, fallback) {
  const raw = args[name];
  if (raw === undefined || raw === null || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid integer for --${name}: ${raw}`);
  }
  return Math.floor(value);
}

function requireArg(args, name) {
  if (!args[name]) throw new Error(`Missing required argument: --${name}`);
  return args[name];
}

function promptCharLimit() {
  const raw = process.env.VC4_GPT_MAX_PROMPT_CHARS || process.env.GPT_WEB_MAX_PROMPT_CHARS || "0";
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid VC4_GPT_MAX_PROMPT_CHARS/GPT_WEB_MAX_PROMPT_CHARS: ${raw}`);
  }
  return Math.floor(value);
}

function enforcePromptCharLimit(promptFile, promptLen, outDir) {
  const limit = promptCharLimit();
  if (limit === 0 || promptLen <= limit) return;
  const message =
    `Prompt is ${promptLen} chars, above safe web-transport limit ${limit}. ` +
    `Slice context is too large for reliable ChatGPT composer submission; reduce selected context or set VC4_GPT_MAX_PROMPT_CHARS=0 to override explicitly.`;
  const diag = {
    ok: false,
    reason: "prompt_too_large_for_web_transport",
    promptFile,
    promptChars: promptLen,
    limitChars: limit,
    overrideEnv: "VC4_GPT_MAX_PROMPT_CHARS=0",
    message,
  };
  try {
    mkdirp(path.join(outDir, ".gpt-web-run"));
    fs.writeFileSync(path.join(outDir, ".gpt-web-run", "prompt_too_large.json"), JSON.stringify(diag, null, 2) + "\n", "utf8");
  } catch (_) {}
  throw new Error(message);
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------
function mkdirp(p) {
  fs.mkdirSync(p, { recursive: true });
}

function randomToken() {
  return `gptweb_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function escapeRegExp(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function stripOuterQuotes(s) {
  const t = String(s || "").trim();
  if (
    (t.startsWith('"') && t.endsWith('"')) ||
    (t.startsWith("'") && t.endsWith("'")) ||
    (t.startsWith("`") && t.endsWith("`"))
  ) {
    return t.slice(1, -1);
  }
  return t;
}

function sha256(text) {
  return crypto.createHash("sha256").update(String(text || ""), "utf8").digest("hex");
}

function readPromptFile(promptFile) {
  const p = path.resolve(promptFile);
  if (!fs.existsSync(p)) throw new Error(`Prompt file does not exist: ${p}`);
  return fs.readFileSync(p, "utf8");
}

function parseKv(header, key) {
  const re = new RegExp(`(?:^|\\s)${escapeRegExp(key)}=(?:"([^"]*)"|'([^']*)'|(\\S+))`);
  const m = String(header || "").match(re);
  if (!m) return null;
  return stripOuterQuotes(m[1] ?? m[2] ?? m[3] ?? "");
}

function normalizeRelativePath(rawPath) {
  let p = stripOuterQuotes(rawPath || "").replace(/\0/g, "").replace(/\\/g, "/").trim();
  while (p.startsWith("./")) p = p.slice(2);
  if (!p) throw new Error("Empty GPTWEB_FILE path");
  if (p.startsWith("/")) throw new Error(`Refusing absolute path: ${p}`);
  const parts = p.split("/").filter((part) => part.length > 0 && part !== ".");
  if (parts.some((part) => part === "..")) throw new Error(`Refusing path with '..': ${p}`);
  const normalized = parts.join("/");
  if (!normalized) throw new Error(`Invalid GPTWEB_FILE path: ${p}`);
  if (normalized === ".gpt-web-run" || normalized.startsWith(".gpt-web-run/")) {
    throw new Error(`Refusing to write into reserved metadata directory: ${normalized}`);
  }
  return normalized;
}

function parseHeaderPath(header) {
  const pathFromKv = parseKv(header, "path");
  if (pathFromKv) return normalizeRelativePath(pathFromKv);
  let h = String(header || "").trim();
  h = h.replace(/(?:^|\s)token=(?:"[^"]*"|'[^']*'|\S+)/g, "").trim();
  h = h.replace(/(?:^|\s)(?:encoding|content_encoding)=(?:"[^"]*"|'[^']*'|\S+)/g, "").trim();
  h = h.replace(/^path=/, "").trim();
  return normalizeRelativePath(h);
}

function headerWantsEntityDecode(header) {
  const enc = String(parseKv(header, "encoding") || parseKv(header, "content_encoding") || "")
    .trim()
    .toLowerCase()
    .replace(/_/g, "-");
  return enc === "html-entities" || enc === "html-entity-shielded" || enc === "entity-shielded";
}

function decodeHtmlEntitiesForFileContent(text) {
  const named = {
    amp: "&",
    lt: "<",
    gt: ">",
    quot: '"',
    apos: "'",
    nbsp: "\u00a0",
  };
  return String(text || "").replace(/&(?:#x([0-9a-fA-F]+)|#([0-9]+)|([A-Za-z][A-Za-z0-9]+));/g, (match, hex, dec, name) => {
    if (hex !== undefined) {
      const cp = Number.parseInt(hex, 16);
      return Number.isFinite(cp) && cp >= 0 && cp <= 0x10ffff ? String.fromCodePoint(cp) : match;
    }
    if (dec !== undefined) {
      const cp = Number.parseInt(dec, 10);
      return Number.isFinite(cp) && cp >= 0 && cp <= 0x10ffff ? String.fromCodePoint(cp) : match;
    }
    const key = String(name || "").toLowerCase();
    return Object.prototype.hasOwnProperty.call(named, key) ? named[key] : match;
  });
}

function headerWantsGitPatchLinesDecode(header) {
  const enc = String(parseKv(header, "encoding") || parseKv(header, "content_encoding") || "")
    .trim()
    .toLowerCase()
    .replace(/_/g, "-");
  return enc === "git-patch-lines" || enc === "patch-lines" || enc === "unified-diff-lines";
}

function decodeGitPatchLinesForFileContent(text) {
  const out = [];
  const lines = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  if (lines.length > 0 && lines[lines.length - 1] === "") {
    lines.pop();
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const m = line.match(/^([RACD])\|(.*)$/);
    if (!m) {
      throw new Error(
        `Invalid git-patch-lines record at encoded line ${i + 1}: expected R|, A|, C|, or D|`
      );
    }

    const kind = m[1];
    const payload = decodeHtmlEntitiesForFileContent(m[2] || "");

    if (kind === "R") {
      out.push(payload);
    } else if (kind === "A") {
      out.push(`+${payload}`);
    } else if (kind === "D") {
      out.push(`-${payload}`);
    } else if (kind === "C") {
      out.push(` ${payload}`);
    }
  }

  return `${out.join("\n")}\n`;
}

function parseFileBlocks(answer, expectedToken) {
  const files = [];
  const text = String(answer || "");
  const re = /^BEGIN_GPT_?WEB_FILE([^\r\n]*)\r?\n([\s\S]*?)^END_GPT_?WEB_FILE([^\r\n]*)(?:\r?\n|$)/gm;
  let m;
  while ((m = re.exec(text)) !== null) {
    const header = String(m[1] || "").trim();
    let content = m[2] ?? "";
    const footer = String(m[3] || "").trim();
    const headerToken = parseKv(header, "token");
    const footerToken = parseKv(footer, "token");
    if (expectedToken) {
      if (headerToken && headerToken !== expectedToken) continue;
      if (footerToken && footerToken !== expectedToken) continue;
    }
    let relPath;
    try {
      relPath = parseHeaderPath(header);
    } catch (err) {
      files.push({ error: err.message, header, skipped: true });
      continue;
    }
    const entityDecoded = headerWantsEntityDecode(header);
    const gitPatchLinesDecoded = headerWantsGitPatchLinesDecode(header);
    try {
      if (gitPatchLinesDecoded) {
        content = decodeGitPatchLinesForFileContent(content);
      } else if (entityDecoded) {
        content = decodeHtmlEntitiesForFileContent(content);
      }
    } catch (err) {
      files.push({ relPath, error: err.message, header, skipped: true });
      continue;
    }
    if (!content.endsWith("\n")) {
      content += "\n";
    }
    files.push({ relPath, content, header, entityDecoded, gitPatchLinesDecoded });
  }
  return files;
}

function writeGeneratedFiles(parsedFiles, outDir) {
  const written = [];
  const skipped = [];
  const fileMetadata = [];
  const root = path.resolve(outDir);
  for (const f of parsedFiles) {
    if (f.skipped) {
      skipped.push(f);
      continue;
    }
    try {
      const relPath = normalizeRelativePath(f.relPath);
      const dest = path.resolve(root, relPath);
      if (!(dest === root || dest.startsWith(root + path.sep))) {
        throw new Error(`Refusing to write outside --out: ${relPath}`);
      }
      mkdirp(path.dirname(dest));
      fs.writeFileSync(dest, String(f.content ?? ""), "utf8");
      written.push(dest);
      fileMetadata.push({
        relPath,
        dest,
        header: String(f.header || ""),
        entityDecoded: !!f.entityDecoded,
        gitPatchLinesDecoded: !!f.gitPatchLinesDecoded,
      });
      vlog("wrote file", { dest, relPath, gitPatchLinesDecoded: !!f.gitPatchLinesDecoded });
    } catch (err) {
      skipped.push({ relPath: f.relPath, error: err.message });
    }
  }
  return { written, skipped, fileMetadata };
}

// ---------------------------------------------------------------------------
// In-flight prompt marker
// ---------------------------------------------------------------------------
function markerPath(repoRoot) {
  return path.join(repoRoot, ".vc4_auto", "in_flight_prompt.json");
}

function readMarker(repoRoot) {
  try {
    const p = markerPath(repoRoot);
    if (!fs.existsSync(p)) return null;
    const raw = fs.readFileSync(p, "utf8");
    const obj = JSON.parse(raw);
    if (!obj || typeof obj !== "object") return null;
    return obj;
  } catch (_) {
    return null;
  }
}

function writeMarker(repoRoot, data) {
  const p = markerPath(repoRoot);
  mkdirp(path.dirname(p));
  const tmp = `${p}.tmp.${process.pid}.${Date.now().toString(36)}`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), "utf8");
  fs.renameSync(tmp, p);
  vlog("marker written", { path: p, data });
}

function clearMarker(repoRoot) {
  try {
    const p = markerPath(repoRoot);
    if (fs.existsSync(p)) {
      fs.unlinkSync(p);
      vlog("marker cleared", { path: p });
    }
  } catch (err) {
    vwarn("clearMarker failed", { err: err.message });
  }
}

// ---------------------------------------------------------------------------
// ChatGPT URL helpers
// ---------------------------------------------------------------------------
function isChatGptUrl(url) {
  return /^https?:\/\/(chatgpt\.com|chat\.openai\.com)(\/|$)/i.test(url || "");
}

function isConversationUrl(url) {
  return /^https?:\/\/(chatgpt\.com|chat\.openai\.com)\/c\/[A-Za-z0-9_-]+/i.test(url || "");
}

function conversationIdFromUrl(url) {
  const m = String(url || "").match(/\/c\/([A-Za-z0-9_-]+)/);
  return m ? m[1] : "";
}

async function waitForConversationUrl(page, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastSeen = page.url();
  while (Date.now() < deadline) {
    const url = page.url();
    if (url !== lastSeen) {
      vlog("tab URL changed", { from: lastSeen, to: url });
      lastSeen = url;
    }
    if (isConversationUrl(url)) return url;
    await page.waitForTimeout(200);
  }
  vwarn("URL did not become a conversation URL", { final: lastSeen, timeoutMs });
  return page.url();
}

async function listChatGptPages(browser) {
  const pages = [];
  for (const context of browser.contexts()) {
    for (const page of context.pages()) {
      const url = page.url();
      if (!isChatGptUrl(url)) continue;
      let title = "";
      try {
        title = await page.title();
      } catch (_) {}
      pages.push({ page, url, title, context });
    }
  }
  return pages;
}

// ---------------------------------------------------------------------------
// Composer / send / generation detection
// ---------------------------------------------------------------------------
const PROMPT_BOX_SELECTORS = [
  // Put the selectors that actually match the current ChatGPT composer first.
  // The previous driver waited 60s on [data-testid="prompt-textarea"] before
  // ever trying #prompt-textarea, which is why every fresh tab burned a minute.
  '#prompt-textarea[contenteditable="true"]',
  'div[contenteditable="true"][id="prompt-textarea"]',
  '.ProseMirror[contenteditable="true"]',
  '[data-testid="prompt-textarea"]',
  '#prompt-textarea',
  'textarea[placeholder*="Message"]',
  'textarea[data-testid="prompt-textarea"]',
  'textarea',
  'div[contenteditable="true"]',
  '[contenteditable="true"]',
];

const SEND_BUTTON_SELECTORS = [
  'button[data-testid="send-button"]',
  'button[aria-label="Send prompt"]',
  'button[aria-label="Send message"]',
  'button[aria-label*="Send"]',
  'form button[type="submit"]',
];

const STOP_BUTTON_SELECTORS = [
  'button[data-testid="stop-button"]',
  'button[aria-label="Stop streaming"]',
  'button[aria-label="Stop generating"]',
  'button[aria-label="Stop generating response"]',
];

const BIG_PROMPT_PASTE_WAIT_MS = Number(process.env.GPT_WEB_BIG_PASTE_WAIT_MS || 15000);
const SMALL_PROMPT_INSERT_TEXT_MAX_CHARS = Number(process.env.GPT_WEB_INSERTTEXT_MAX_CHARS || 20000);

function promptLoadThreshold(expectedLen) {
  if (!expectedLen || expectedLen < 10000) return 1;
  // textContent can differ from the source string because contenteditable
  // paragraphs add/drop line separators.  85% is strict enough to reject a
  // one-character/partial paste but tolerant of DOM normalization.
  return Math.floor(expectedLen * 0.85);
}

async function findPromptBox(page, timeout = 10000) {
  const startedAt = Date.now();
  const handle = await page
    .waitForFunction(
      ({ selectors }) => {
        function isCandidateVisible(el) {
          if (!el) return false;
          const style = window.getComputedStyle(el);
          if (style.visibility === "hidden" || style.display === "none") return false;
          const rect = el.getBoundingClientRect();
          if (rect.width <= 0 || rect.height <= 0) return false;
          if (el.closest('[aria-hidden="true"]')) return false;
          const tag = el.tagName;
          return el.isContentEditable || tag === "TEXTAREA" || tag === "INPUT";
        }
        for (const selector of selectors) {
          const nodes = Array.from(document.querySelectorAll(selector));
          for (let i = nodes.length - 1; i >= 0; i--) {
            const el = nodes[i];
            if (isCandidateVisible(el)) return el;
          }
        }
        return null;
      },
      { selectors: PROMPT_BOX_SELECTORS },
      { timeout, polling: 100 }
    )
    .catch((err) => {
      throw new Error(`Could not find ChatGPT prompt box within ${timeout}ms: ${err.message}`);
    });
  const el = handle.asElement();
  if (!el) throw new Error("Could not find ChatGPT prompt box: waitForFunction returned no element");
  const desc = await el
    .evaluate((node) => {
      const rect = node.getBoundingClientRect();
      return {
        tag: node.tagName.toLowerCase(),
        id: node.id || "",
        testid: node.getAttribute("data-testid") || "",
        className: String(node.getAttribute("class") || "").slice(0, 80),
        elapsedMs: 0,
        rect: {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          w: Math.round(rect.width),
          h: Math.round(rect.height),
        },
      };
    })
    .catch(() => ({}));
  desc.elapsedMs = Date.now() - startedAt;
  vlog("findPromptBox: matched", desc);
  return el;
}

async function composerTextLength(page) {
  return await page
    .evaluate((selectors) => {
      function visibleEditable(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return false;
        const tag = el.tagName;
        return el.isContentEditable || tag === "TEXTAREA" || tag === "INPUT";
      }
      for (const selector of selectors) {
        const nodes = Array.from(document.querySelectorAll(selector));
        for (let i = nodes.length - 1; i >= 0; i--) {
          const el = nodes[i];
          if (!visibleEditable(el)) continue;
          if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") return (el.value || "").length;
          // textContent is much cheaper than innerText for 500k+ character
          // prompts because it does not force layout.
          return (el.textContent || "").length;
        }
      }
      return 0;
    }, PROMPT_BOX_SELECTORS)
    .catch(() => 0);
}


function hasPromptMaterialState(state) {
  return !!(state && (state.composerLen > 0 || state.attachmentCount > 0 || state.showInTextFieldCount > 0));
}

// ChatGPT can convert a huge rich paste into an attachment chip with a
// "Show in text field" affordance.  The submitter must treat that as prompt
// material rather than as a failed paste, and it should try to expand the chip
// into the text field before falling back to attachment submission.
async function promptMaterialState(page) {
  return await page
    .evaluate((selectors) => {
      function visible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }
      function findComposer() {
        for (const selector of selectors) {
          const nodes = Array.from(document.querySelectorAll(selector));
          for (let i = nodes.length - 1; i >= 0; i--) {
            const el = nodes[i];
            const tag = el.tagName;
            if (visible(el) && (el.isContentEditable || tag === "TEXTAREA" || tag === "INPUT")) return el;
          }
        }
        return null;
      }
      function labelFor(el) {
        return [
          el.getAttribute("aria-label") || "",
          el.getAttribute("title") || "",
          el.getAttribute("data-testid") || "",
          el.getAttribute("class") || "",
          ((el.textContent || "") + "").trim(),
        ].join(" ").replace(/\s+/g, " ").trim();
      }

      const composer = findComposer();
      const root =
        (composer && (composer.closest("form") || composer.closest('[data-testid*="composer" i]'))) ||
        (composer && composer.parentElement && composer.parentElement.parentElement) ||
        document.body;
      const state = {
        composerLen: 0,
        composerTag: composer ? composer.tagName.toLowerCase() : "",
        composerId: composer ? composer.id || "" : "",
        attachmentCount: 0,
        showInTextFieldCount: 0,
        attachmentHints: [],
        showHints: [],
      };
      if (composer) {
        state.composerLen =
          composer.tagName === "TEXTAREA" || composer.tagName === "INPUT"
            ? (composer.value || "").length
            : (composer.textContent || "").length;
      }

      const showRe = /(?:^|\b)(?:show|insert)\s+(?:in|as|into)\s+(?:the\s+)?(?:text\s+field|composer)\b/i;
      const showBlockedRe = /too\s+long\s+to\s+show\s+in\s+(?:the\s+)?text\s+field|open\s+pasted\s+text\s+attachment/i;
      const removeFileRe = /(?:remove|delete|discard|detach).{0,40}(?:file|attachment|upload)|(?:file|attachment|upload).{0,40}(?:remove|delete|discard|detach)/i;
      const attachmentClassRe = /attachment|uploaded[-_ ]?file|file[-_ ]?chip|upload[-_ ]?preview|composer[-_ ]?file|file[-_ ]?preview/i;
      const attachmentTextRe = /attached\s+file|uploaded\s+file|file\s+attached|attachment|show\s+in\s+text\s+field|\.txt\b|\.md\b|\.json\b/i;
      const skipRe = /add\s+files\s+and\s+more|start\s+a\s+group\s+chat|turn\s+on\s+temporary\s+chat|open\s+sidebar|close\s+sidebar|model\s*$/i;

      const nodes = Array.from(root.querySelectorAll('button, [role="button"], [role="menuitem"], a, [aria-label], [data-testid], [class]'));
      const seen = new Set();
      for (const el of nodes) {
        if (seen.has(el) || !visible(el)) continue;
        seen.add(el);
        if (composer && el === composer) continue;
        const blob = labelFor(el);
        if (!blob || skipRe.test(blob)) continue;
        const isShowInTextFieldAction = showRe.test(blob) && !showBlockedRe.test(blob);
        if (isShowInTextFieldAction) {
          state.showInTextFieldCount++;
          if (state.showHints.length < 8) state.showHints.push(blob.slice(0, 160));
        }
        if (isShowInTextFieldAction || removeFileRe.test(blob) || attachmentClassRe.test(blob) || attachmentTextRe.test(blob)) {
          state.attachmentCount++;
          if (state.attachmentHints.length < 12) state.attachmentHints.push(blob.slice(0, 160));
        }
      }
      return state;
    }, PROMPT_BOX_SELECTORS)
    .catch((err) => ({
      composerLen: 0,
      composerTag: "",
      composerId: "",
      attachmentCount: 0,
      showInTextFieldCount: 0,
      attachmentHints: [],
      showHints: [],
      error: String(err),
    }));
}

async function tryShowAttachmentInTextField(page) {
  async function clickVisibleShowButton(phase) {
    return await page
      .evaluate(({ selectors, phase }) => {
        function visible(el) {
          if (!el) return false;
          const style = window.getComputedStyle(el);
          if (style.visibility === "hidden" || style.display === "none") return false;
          const rect = el.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        }
        function findComposer() {
          for (const selector of selectors) {
            const nodes = Array.from(document.querySelectorAll(selector));
            for (let i = nodes.length - 1; i >= 0; i--) {
              const el = nodes[i];
              const tag = el.tagName;
              if (visible(el) && (el.isContentEditable || tag === "TEXTAREA" || tag === "INPUT")) return el;
            }
          }
          return null;
        }
        function labelFor(el) {
          return [
            el.getAttribute("aria-label") || "",
            el.getAttribute("title") || "",
            el.getAttribute("data-testid") || "",
            ((el.textContent || "") + "").trim(),
          ].join(" ").replace(/\s+/g, " ").trim();
        }
        const composer = findComposer();
        const root =
          (composer && (composer.closest("form") || composer.closest('[data-testid*="composer" i]'))) ||
          (composer && composer.parentElement && composer.parentElement.parentElement) ||
          document.body;
        const showRe = /(?:^|\b)(?:show|insert)\s+(?:in|as|into)\s+(?:the\s+)?(?:text\s+field|composer)\b/i;
        const showBlockedRe = /too\s+long\s+to\s+show\s+in\s+(?:the\s+)?text\s+field|open\s+pasted\s+text\s+attachment/i;
        const nodes = Array.from(root.querySelectorAll('button, [role="button"], [role="menuitem"], a'));
        for (const el of nodes) {
          if (!visible(el)) continue;
          const blob = labelFor(el);
          if (!showRe.test(blob) || showBlockedRe.test(blob)) continue;
          el.click();
          return { ok: true, phase, blob: blob.slice(0, 180) };
        }
        return { ok: false, phase, reason: "no visible Show in text field control" };
      }, { selectors: PROMPT_BOX_SELECTORS, phase })
      .catch((err) => ({ ok: false, phase, reason: String(err) }));
  }

  let result = await clickVisibleShowButton("direct");
  if (result.ok) {
    vlog("tryShowAttachmentInTextField: clicked direct control", result);
    await page.waitForTimeout(1000);
    return true;
  }

  const menuResult = await page
    .evaluate((selectors) => {
      function visible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }
      function findComposer() {
        for (const selector of selectors) {
          const nodes = Array.from(document.querySelectorAll(selector));
          for (let i = nodes.length - 1; i >= 0; i--) {
            const el = nodes[i];
            const tag = el.tagName;
            if (visible(el) && (el.isContentEditable || tag === "TEXTAREA" || tag === "INPUT")) return el;
          }
        }
        return null;
      }
      function labelFor(el) {
        return [
          el.getAttribute("aria-label") || "",
          el.getAttribute("title") || "",
          el.getAttribute("data-testid") || "",
          ((el.textContent || "") + "").trim(),
        ].join(" ").replace(/\s+/g, " ").trim();
      }
      const composer = findComposer();
      const root =
        (composer && (composer.closest("form") || composer.closest('[data-testid*="composer" i]'))) ||
        (composer && composer.parentElement && composer.parentElement.parentElement) ||
        document.body;
      const menuRe = /(?:attachment|file|upload|more|options|menu|\.txt\b|\.md\b|\.json\b)/i;
      const skipRe = /add\s+files\s+and\s+more|send\s+(prompt|message)|model|start\s+voice|start\s+dictation|open\s+sidebar|close\s+sidebar/i;
      const nodes = Array.from(root.querySelectorAll('button, [role="button"], [aria-haspopup="menu"]'));
      for (const el of nodes) {
        if (!visible(el)) continue;
        const blob = labelFor(el);
        if (skipRe.test(blob)) continue;
        if (!menuRe.test(blob)) continue;
        el.click();
        return { ok: true, blob: blob.slice(0, 180) };
      }
      return { ok: false, reason: "no likely attachment menu" };
    }, PROMPT_BOX_SELECTORS)
    .catch((err) => ({ ok: false, reason: String(err) }));
  vlog("tryShowAttachmentInTextField: menu probe", menuResult);
  if (menuResult.ok) {
    await page.waitForTimeout(350);
    result = await clickVisibleShowButton("after-menu");
    if (result.ok) {
      vlog("tryShowAttachmentInTextField: clicked after opening menu", result);
      await page.waitForTimeout(1000);
      return true;
    }
  }
  vlog("tryShowAttachmentInTextField: no show-in-text-field action taken", result);
  return false;
}

async function clearPromptBoxInDom(page) {
  await page
    .evaluate((selectors) => {
      function visibleEditable(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return false;
        const tag = el.tagName;
        return el.isContentEditable || tag === "TEXTAREA" || tag === "INPUT";
      }
      for (const selector of selectors) {
        const nodes = Array.from(document.querySelectorAll(selector));
        for (let i = nodes.length - 1; i >= 0; i--) {
          const el = nodes[i];
          if (!visibleEditable(el)) continue;
          el.focus();
          if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
            el.value = "";
            el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "deleteContentBackward" }));
            return true;
          }
          const sel = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(el);
          sel.removeAllRanges();
          sel.addRange(range);
          try {
            document.execCommand("delete", false, null);
          } catch (_) {
            el.textContent = "";
          }
          el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "deleteContentBackward" }));
          return true;
        }
      }
      return false;
    }, PROMPT_BOX_SELECTORS)
    .catch(() => {});
}

async function assistantCount(page) {
  return await page.locator('[data-message-author-role="assistant"]').count().catch(() => 0);
}

async function lastAssistantText(page) {
  const loc = page.locator('[data-message-author-role="assistant"]');
  const count = await loc.count().catch(() => 0);
  if (count > 0) {
    const text = await loc.nth(count - 1).innerText().catch(() => "");
    if (text.trim()) return text.trim();
  }
  const markdown = page.locator(".markdown");
  const mdCount = await markdown.count().catch(() => 0);
  if (mdCount > 0) {
    const text = await markdown.nth(mdCount - 1).innerText().catch(() => "");
    if (text.trim()) return text.trim();
  }
  return "";
}

// Conservative: only Stop button or thinking/reasoning data-testid.  No
// free-text matching (which previously caused infinite waits because of
// homepage chrome).
async function isGenerating(page) {
  return await page
    .evaluate((stopSelectors) => {
      function visible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }
      for (const selector of stopSelectors) {
        const nodes = Array.from(document.querySelectorAll(selector));
        if (nodes.some(visible)) return true;
      }
      const thinking = document.querySelectorAll('[data-testid*="thinking" i], [data-testid*="reasoning" i]');
      if (Array.from(thinking).some(visible)) return true;
      return false;
    }, STOP_BUTTON_SELECTORS)
    .catch(() => false);
}

async function inspectSendButton(page) {
  return await page
    .evaluate((selectors) => {
      function visible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }
      const seen = new Set();
      const candidates = [];
      for (const selector of selectors) {
        for (const b of document.querySelectorAll(selector)) {
          if (!seen.has(b)) {
            seen.add(b);
            candidates.push({ button: b, selector });
          }
        }
      }
      for (const { button: b, selector } of candidates.reverse()) {
        if (!visible(b)) continue;
        const aria = b.getAttribute("aria-label") || "";
        const testid = b.getAttribute("data-testid") || "";
        const txt = ((b.textContent || "") + "").trim();
        if (/stop/i.test(aria) || /stop/i.test(testid) || /^stop$/i.test(txt)) continue;
        const ariaDisabled = b.getAttribute("aria-disabled") || "";
        const disabled = !!b.disabled || ariaDisabled === "true";
        return {
          selector,
          visible: true,
          disabled,
          aria,
          testid,
          ariaDisabled,
          textPreview: txt.slice(0, 40),
        };
      }
      return null;
    }, SEND_BUTTON_SELECTORS)
    .catch(() => null);
}

async function clickSendButton(page) {
  return await page
    .evaluate((selectors) => {
      function visible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }
      const seen = new Set();
      const candidates = [];
      for (const selector of selectors) {
        for (const b of document.querySelectorAll(selector)) {
          if (!seen.has(b)) {
            seen.add(b);
            candidates.push({ button: b, selector });
          }
        }
      }
      for (const { button: b, selector } of candidates.reverse()) {
        if (!visible(b)) continue;
        const aria = b.getAttribute("aria-label") || "";
        const testid = b.getAttribute("data-testid") || "";
        if (/stop/i.test(aria) || /stop/i.test(testid)) continue;
        const ariaDisabled = b.getAttribute("aria-disabled") || "";
        const disabled = !!b.disabled || ariaDisabled === "true";
        if (disabled) continue;
        b.click();
        return { ok: true, selector, aria, testid };
      }
      return { ok: false, reason: "no enabled visible send button" };
    }, SEND_BUTTON_SELECTORS)
    .catch((err) => ({ ok: false, reason: String(err) }));
}

async function findSendButton(page) {
  return await inspectSendButton(page);
}

async function waitForComposerReadyForNextPrompt(page, timeoutMs = 120000, label = "next prompt") {
  vlog("waitForComposerReadyForNextPrompt: start", { label, timeoutMs });
  const deadline = Date.now() + timeoutMs;
  let lastLog = 0;
  let iter = 0;
  while (Date.now() < deadline) {
    iter++;
    const generating = await isGenerating(page);
    let composerVisible = false;
    try {
      await findPromptBox(page, 750);
      composerVisible = true;
    } catch (_) {}
    if (composerVisible && !generating) {
      vlog("waitForComposerReadyForNextPrompt: ready", { label, iter });
      return true;
    }
    if (Date.now() - lastLog > 5000) {
      vlog("waitForComposerReadyForNextPrompt: still waiting", {
        label,
        iter,
        composerVisible,
        generating,
        remainingMs: Math.max(0, deadline - Date.now()),
      });
      lastLog = Date.now();
    }
    await page.waitForTimeout(750);
  }
  vwarn("waitForComposerReadyForNextPrompt: timed out", { label, timeoutMs });
  return false;
}

// ---------------------------------------------------------------------------
// Clipboard / paste
// ---------------------------------------------------------------------------
function setClipboardText(text) {
  const maxBuffer = Math.max(64 * 1024 * 1024, Buffer.byteLength(String(text || ""), "utf8") + 1024);
  if (process.platform === "darwin") {
    child_process.execFileSync("/usr/bin/pbcopy", { input: text, maxBuffer });
    return;
  }
  if (process.env.WAYLAND_DISPLAY) {
    child_process.execFileSync("wl-copy", [], { input: text, maxBuffer });
    return;
  }
  child_process.execFileSync("xclip", ["-selection", "clipboard"], { input: text, maxBuffer });
}

function nativePlainTextPaste(usePlain = false) {
  if (process.platform === "darwin") {
    const modifiers = usePlain ? "{command down, shift down}" : "{command down}";
    const script = [
      'tell application "System Events"',
      `  keystroke "v" using ${modifiers}`,
      "end tell",
    ].join("\n");
    child_process.execFileSync("/usr/bin/osascript", ["-e", script], { stdio: "ignore" });
    return;
  }
  throw new Error("native paste is only implemented through osascript on macOS");
}

// Make Chrome the frontmost app on macOS.  Required before sending native
// keystrokes via osascript -- otherwise Cmd+V goes to Terminal.  The primary
// path below uses Playwright keyboard events and normally does not need this;
// it is retained only as an OS-level fallback.
function activateChromeOnMac() {
  if (process.platform !== "darwin") return false;
  for (const appName of ["Google Chrome", "Chromium", "Google Chrome Beta", "Google Chrome Canary"]) {
    try {
      child_process.execFileSync("/usr/bin/osascript", ["-e", `tell application "${appName}" to activate`], {
        stdio: "ignore",
      });
      return true;
    } catch (_) {}
  }
  return false;
}

async function grantClipboardPermissions(page) {
  try {
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], {
      origin: "https://chatgpt.com",
    });
  } catch (_) {}
  try {
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], {
      origin: "https://chat.openai.com",
    });
  } catch (_) {}
}

async function focusPromptBox(page, timeout = 5000) {
  const box = await findPromptBox(page, timeout);
  // Avoid Playwright's default 30s action timeout here; a visible composer is
  // normally already in view, and focus via DOM is enough for the paste paths.
  await box.evaluate((el) => el.focus()).catch(() => {});
  await box.scrollIntoViewIfNeeded({ timeout: 1000 }).catch(() => {});
  await box.click({ timeout: Math.min(timeout, 1500), force: true }).catch(async () => {
    await box.evaluate((el) => el.focus()).catch(() => {});
  });
  return box;
}

async function waitForPromptLoaded(page, expectedLen, timeoutMs = BIG_PROMPT_PASTE_WAIT_MS) {
  const deadline = Date.now() + timeoutMs;
  const threshold = promptLoadThreshold(expectedLen);
  const startedAt = Date.now();
  let lastLen = 0;
  let lastState = null;
  let lastLog = 0;
  let sawAttachment = false;
  let triedShow = false;
  while (Date.now() < deadline) {
    const state = await promptMaterialState(page);
    lastState = state;
    const len = state.composerLen || 0;
    if (len >= threshold) return { ok: true, len, threshold, material: state };
    if (len > 0 && expectedLen < 10000) return { ok: true, len, threshold, material: state };
    lastLen = len;

    if (!triedShow && state.showInTextFieldCount > 0) {
      sawAttachment = true;
      triedShow = true;
      vlog("waitForPromptLoaded: attachment has Show in text field; clicking it", {
        len,
        showInTextFieldCount: state.showInTextFieldCount,
        attachmentHints: state.attachmentHints,
      });
      await tryShowAttachmentInTextField(page);
      await page.waitForTimeout(700);
      continue;
    }

    if (state.attachmentCount > 0) {
      sawAttachment = true;
      // An attachment chip is real prompt material.  Do not burn the entire
      // paste timeout waiting for textbox length to change; the submit loop is
      // attachment-aware and can click Show in text field or submit if enabled.
      if (Date.now() - startedAt > 1800) {
        return { ok: false, len, threshold, attachment: true, material: state };
      }
    }

    if (Date.now() - lastLog > 3000) {
      vlog("waitForPromptLoaded: waiting", {
        len,
        threshold,
        expectedLen,
        attachmentCount: state.attachmentCount,
        showInTextFieldCount: state.showInTextFieldCount,
        remainingMs: Math.max(0, deadline - Date.now()),
      });
      lastLog = Date.now();
    }
    await page.waitForTimeout(250);
  }
  return { ok: false, len: lastLen, threshold, attachment: sawAttachment, material: lastState };
}

async function playwrightClipboardKeyPaste(page, fullPrompt) {
  // Plain-text paste first.  On current ChatGPT, normal Cmd/Ctrl+V can turn a
  // very large clipboard payload into an attachment chip.  Cmd/Ctrl+Shift+V is
  // the accessibility-friendly path: it behaves like "paste as plain text" and
  // lands in the actual text field.  Rich paste is opt-in because it is the
  // path that created the attachment-only state in your log.
  const allowRichPaste = /^(1|true|yes)$/i.test(process.env.GPT_WEB_ALLOW_RICH_PASTE || "");
  const combos = process.platform === "darwin" ? ["Meta+Shift+V"] : ["Control+Shift+V"];
  if (allowRichPaste) combos.push(process.platform === "darwin" ? "Meta+V" : "Control+V");

  for (const combo of combos) {
    vlog("playwrightClipboardKeyPaste: pressing paste shortcut", { combo, len: fullPrompt.length });
    await focusPromptBox(page, 5000).catch(() => {});
    try {
      await page.keyboard.press(combo);
    } catch (err) {
      vwarn("playwrightClipboardKeyPaste: keyboard paste failed", { combo, err: err.message });
      continue;
    }
    const loaded = await waitForPromptLoaded(page, fullPrompt.length, BIG_PROMPT_PASTE_WAIT_MS);
    const material = loaded.material || (await promptMaterialState(page));
    vlog("playwrightClipboardKeyPaste: load result", { combo, ...loaded, material });
    if (loaded.ok || hasPromptMaterialState(material)) return true;
    await clearPromptBoxInDom(page);
  }
  return false;
}

// Dispatch a ClipboardEvent directly at the focused composer.  This is not
// the first choice because some ProseMirror builds intentionally ignore
// synthetic clipboard events, but it is fast and independent of OS focus.
async function playwrightPaste(page, fullPrompt) {
  vlog("playwrightPaste: dispatching synthetic paste event", { len: fullPrompt.length });
  const result = await page
    .evaluate(
      ({ text, selectors }) => {
        function findEl() {
          for (const selector of selectors) {
            const nodes = Array.from(document.querySelectorAll(selector));
            for (let i = nodes.length - 1; i >= 0; i--) {
              const el = nodes[i];
              const style = window.getComputedStyle(el);
              const rect = el.getBoundingClientRect();
              if (style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0) {
                return el;
              }
            }
          }
          return null;
        }
        const el = findEl();
        if (!el) return { ok: false, reason: "no composer element found" };
        el.focus();
        const dt = new DataTransfer();
        dt.setData("text/plain", text);
        const evt = new ClipboardEvent("paste", { bubbles: true, cancelable: true, clipboardData: dt });
        try {
          Object.defineProperty(evt, "clipboardData", { value: dt });
        } catch (_) {}
        const dispatched = el.dispatchEvent(evt);
        if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
          if (!el.value) {
            el.value = text;
            el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertFromPaste", data: text.slice(0, 1) }));
          }
        }
        const len = el.tagName === "TEXTAREA" || el.tagName === "INPUT" ? (el.value || "").length : (el.textContent || "").length;
        return { ok: len > 0, len, dispatched, tag: el.tagName.toLowerCase() };
      },
      { text: fullPrompt, selectors: PROMPT_BOX_SELECTORS }
    )
    .catch((err) => ({ ok: false, reason: String(err) }));
  vlog("playwrightPaste: result", result);
  return !!(result && result.ok);
}

// Direct DOM fallback.  This is intentionally fast: it avoids execCommand on
// very large prompts unless explicitly requested through GPT_WEB_USE_EXEC_INSERT=1.
async function domInsertPrompt(page, fullPrompt) {
  const result = await page
    .evaluate(
      ({ text, selectors, useExec }) => {
        function findEl() {
          for (const selector of selectors) {
            const nodes = Array.from(document.querySelectorAll(selector));
            for (let i = nodes.length - 1; i >= 0; i--) {
              const el = nodes[i];
              const style = window.getComputedStyle(el);
              const rect = el.getBoundingClientRect();
              if (style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0) return el;
            }
          }
          return null;
        }
        const el = findEl();
        if (!el) return { ok: false, reason: "no composer element" };
        el.focus();
        if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
          el.value = text;
          el.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, inputType: "insertFromPaste", data: text.slice(0, 1) }));
          el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertFromPaste", data: text.slice(0, 1) }));
          el.dispatchEvent(new Event("change", { bubbles: true }));
          return { ok: true, len: (el.value || "").length, mode: "textarea-value" };
        }
        const sel = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(el);
        sel.removeAllRanges();
        sel.addRange(range);
        try {
          document.execCommand("delete", false, null);
        } catch (_) {
          el.textContent = "";
        }
        if (useExec) {
          const ok = document.execCommand("insertText", false, text);
          el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text.slice(0, 1) }));
          return { ok: ok || (el.textContent || "").length > 0, len: (el.textContent || "").length, mode: "execCommand" };
        }
        el.textContent = text;
        // Place the caret at the end so ChatGPT sees a normal focused editor.
        const endRange = document.createRange();
        endRange.selectNodeContents(el);
        endRange.collapse(false);
        sel.removeAllRanges();
        sel.addRange(endRange);
        el.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, inputType: "insertFromPaste", data: text.slice(0, 1) }));
        el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertFromPaste", data: text.slice(0, 1) }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return { ok: (el.textContent || "").length > 0, len: (el.textContent || "").length, mode: "textContent" };
      },
      { text: fullPrompt, selectors: PROMPT_BOX_SELECTORS, useExec: /^(1|true|yes)$/i.test(process.env.GPT_WEB_USE_EXEC_INSERT || "") }
    )
    .catch((err) => ({ ok: false, reason: String(err) }));
  vlog("domInsertPrompt: result", result);
  return !!(result && result.ok);
}

async function typePrompt(page, fullPrompt) {
  if (fullPrompt.length > SMALL_PROMPT_INSERT_TEXT_MAX_CHARS) {
    vwarn("typePrompt: refusing slow character-by-character fallback for large prompt", {
      len: fullPrompt.length,
      max: SMALL_PROMPT_INSERT_TEXT_MAX_CHARS,
    });
    return false;
  }
  vlog("typePrompt: focusing composer and typing", { len: fullPrompt.length });
  try {
    await focusPromptBox(page, 5000);
    await page.keyboard.type(fullPrompt, { delay: 0 });
    return true;
  } catch (err) {
    vwarn("typePrompt: type() failed", { err: err.message });
    return false;
  }
}

async function sendViaKeyboard(page, allowBareEnter = false) {
  try {
    await focusPromptBox(page, 5000);
  } catch (err) {
    vwarn("sendViaKeyboard: could not focus composer", { err: err.message });
    return false;
  }
  const combos = allowBareEnter
    ? ["Enter"]
    : process.platform === "darwin"
      ? ["Meta+Enter"]
      : ["Control+Enter"];
  for (const combo of combos) {
    try {
      vlog("sendViaKeyboard: pressing", { combo });
      await page.keyboard.press(combo);
      await page.waitForTimeout(400);
      return true;
    } catch (err) {
      vwarn("sendViaKeyboard: combo failed", { combo, err: err.message });
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Composer state dump.  Pulled from the page on every important event so
// the log captures exactly what the script was looking at.
// ---------------------------------------------------------------------------
async function dumpComposerState(page, label) {
  const info = await page
    .evaluate((selectors) => {
      const out = { url: location.href, candidates: [], buttons: [], attachmentLike: [] };
      for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (!el) continue;
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        const visible = style.visibility !== "hidden" && style.display !== "none";
        const text = el.tagName === "TEXTAREA" || el.tagName === "INPUT" ? el.value || "" : el.textContent || "";
        out.candidates.push({
          selector,
          tag: el.tagName.toLowerCase(),
          id: el.id || "",
          visible,
          textLen: text.length,
          textPreview: text.slice(0, 80),
          rect: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            w: Math.round(rect.width),
            h: Math.round(rect.height),
          },
        });
      }
      const buttons = document.querySelectorAll('button[data-testid], button[aria-label], button[type="submit"], form button');
      for (const b of buttons) {
        const style = window.getComputedStyle(b);
        const rect = b.getBoundingClientRect();
        const visible = style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
        if (!visible) continue;
        out.buttons.push({
          testid: b.getAttribute("data-testid") || "",
          aria: b.getAttribute("aria-label") || "",
          type: b.getAttribute("type") || "",
          disabled: !!b.disabled,
          ariaDisabled: b.getAttribute("aria-disabled") || "",
          textPreview: ((b.textContent || "") + "").trim().slice(0, 60),
        });
      }
      const showRe = /(?:^|\b)(?:show|insert)\s+(?:in|as|into)\s+(?:the\s+)?(?:text\s+field|composer)\b/i;
      const showBlockedRe = /too\s+long\s+to\s+show\s+in\s+(?:the\s+)?text\s+field|open\s+pasted\s+text\s+attachment/i;
      const attachRe = /attachment|attached|uploaded[-_ ]?file|file[-_ ]?chip|upload[-_ ]?preview|show\s+in\s+text\s+field|remove.{0,30}(file|attachment)/i;
      for (const el of document.querySelectorAll('button, [role="button"], [role="menuitem"], [aria-label], [data-testid], [class]')) {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        const visible = style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
        if (!visible) continue;
        const blob = [
          el.getAttribute("aria-label") || "",
          el.getAttribute("title") || "",
          el.getAttribute("data-testid") || "",
          el.getAttribute("class") || "",
          ((el.textContent || "") + "").trim(),
        ].join(" ").replace(/\s+/g, " ").trim();
        const isShowInTextFieldAction = showRe.test(blob) && !showBlockedRe.test(blob);
        if ((isShowInTextFieldAction || attachRe.test(blob)) && !/add\s+files\s+and\s+more/i.test(blob)) {
          out.attachmentLike.push(blob.slice(0, 160));
          if (out.attachmentLike.length >= 16) break;
        }
      }
      return out;
    }, PROMPT_BOX_SELECTORS)
    .catch((err) => ({ error: String(err) }));
  vlog(`dumpComposerState @ ${label}`, info);
  return info;
}

// ---------------------------------------------------------------------------
// Submit core
//
// The fast path is OS clipboard + Playwright plain-text keyboard paste
// (Cmd/Ctrl+Shift+V).  It is much faster for 500k+ character prompts than
// keyboard.insertText and avoids ChatGPT's attachment conversion for normal
// rich paste.  Fallback order: synthetic ClipboardEvent, native macOS plain
// paste, direct DOM set, and only for small prompts keyboard.insertText/type.
// ---------------------------------------------------------------------------
async function submitPrompt(page, fullPrompt, promptTimeoutMs) {
  vlog("submitPrompt: start", { len: fullPrompt.length, promptTimeoutMs });
  await page.bringToFront().catch(() => {});
  await grantClipboardPermissions(page);

  vlog("submitPrompt: locating/focusing prompt box");
  await focusPromptBox(page, Math.min(promptTimeoutMs, 15000));
  vlog("submitPrompt: clearing composer");
  await clearPromptBoxInDom(page);
  await page.waitForTimeout(100);

  const beforeCount = await assistantCount(page);
  const insertStartedAt = Date.now();
  const threshold = promptLoadThreshold(fullPrompt.length);
  const strategyResults = [];
  let materialState = await promptMaterialState(page);
  let lenAfter = materialState.composerLen || 0;

  async function refreshMaterial(label) {
    materialState = await promptMaterialState(page);
    lenAfter = materialState.composerLen || 0;
    strategyResults.push(
      `${label}:${lenAfter}:attachments=${materialState.attachmentCount}:show=${materialState.showInTextFieldCount}`
    );
    return materialState;
  }

  async function showAttachmentIfAvailable(label) {
    const state = await promptMaterialState(page);
    if (state.showInTextFieldCount <= 0) return false;
    vlog("submitPrompt: Show in text field available; clicking", {
      label,
      composerLen: state.composerLen,
      attachmentCount: state.attachmentCount,
      showInTextFieldCount: state.showInTextFieldCount,
      attachmentHints: state.attachmentHints,
    });
    const clicked = await tryShowAttachmentInTextField(page);
    await refreshMaterial(`${label}:showClicked=${clicked}`);
    return clicked;
  }

  vlog("submitPrompt: preloading OS clipboard");
  try {
    setClipboardText(fullPrompt);
    strategyResults.push("clipboard:set-ok");
  } catch (err) {
    strategyResults.push(`clipboard:set-failed:${err.message}`);
    vwarn("submitPrompt: clipboard set failed", { err: err.message });
  }

  // Strategy 1: plain-text browser paste via Playwright keyboard events.
  // Normal Cmd/Ctrl+V is intentionally opt-in because it can create attachment
  // chips for huge payloads.  Cmd/Ctrl+Shift+V is the accessibility-oriented
  // text-field path.
  let keyPasteOk = false;
  if (!hasPromptMaterialState(materialState)) {
    keyPasteOk = await playwrightClipboardKeyPaste(page, fullPrompt);
    await refreshMaterial(`keyPaste:${keyPasteOk}`);
    if (lenAfter < threshold) await showAttachmentIfAvailable("after-key-paste");
  }

  // Strategy 2: synthetic ClipboardEvent fallback.  Only run if the page still
  // has no prompt material at all; if ChatGPT already created an attachment chip
  // then the submit loop can handle it without deleting the material.
  let pasteOk = false;
  if (lenAfter < threshold && !hasPromptMaterialState(materialState)) {
    await clearPromptBoxInDom(page);
    pasteOk = await playwrightPaste(page, fullPrompt);
    const loaded = await waitForPromptLoaded(page, fullPrompt.length, 10000);
    materialState = loaded.material || (await promptMaterialState(page));
    lenAfter = Math.max(loaded.len || 0, materialState.composerLen || 0);
    strategyResults.push(`syntheticPaste:${pasteOk}:${lenAfter}:attachments=${materialState.attachmentCount}`);
    if (lenAfter < threshold) await showAttachmentIfAvailable("after-synthetic-paste");
  }

  // Strategy 3: native macOS paste after activating Chrome.  Plain-text
  // Cmd+Shift+V is tried before rich Cmd+V.
  if (lenAfter < threshold && !hasPromptMaterialState(materialState) && process.platform === "darwin") {
    await clearPromptBoxInDom(page);
    vlog("submitPrompt: trying OS-level paste fallback after activating Chrome");
    const activated = activateChromeOnMac();
    vlog("submitPrompt: activated Chrome.app", { activated });
    await page.bringToFront().catch(() => {});
    await focusPromptBox(page, 5000).catch(() => {});
    for (const usePlain of [true, false]) {
      try {
        nativePlainTextPaste(usePlain);
      } catch (err) {
        vwarn("submitPrompt: native paste failed", { usePlain, err: err.message });
      }
      const loaded = await waitForPromptLoaded(page, fullPrompt.length, 15000);
      materialState = loaded.material || (await promptMaterialState(page));
      lenAfter = Math.max(loaded.len || 0, materialState.composerLen || 0);
      strategyResults.push(`nativePaste:${usePlain ? "plain" : "normal"}:${lenAfter}:attachments=${materialState.attachmentCount}`);
      if (lenAfter < threshold) await showAttachmentIfAvailable(`after-native-${usePlain ? "plain" : "normal"}`);
      if (lenAfter >= threshold || hasPromptMaterialState(materialState)) break;
      await clearPromptBoxInDom(page);
    }
  }

  // Strategy 4: direct DOM insertion.  Fast, but less preferred because it
  // bypasses the normal paste path.  Useful when browser paste is blocked.
  let domInsertOk = false;
  if (lenAfter < threshold && !hasPromptMaterialState(materialState)) {
    await clearPromptBoxInDom(page);
    domInsertOk = await domInsertPrompt(page, fullPrompt);
    const loaded = await waitForPromptLoaded(page, fullPrompt.length, 5000);
    materialState = loaded.material || (await promptMaterialState(page));
    lenAfter = Math.max(loaded.len || 0, materialState.composerLen || 0);
    strategyResults.push(`domInsert:${domInsertOk}:${lenAfter}:attachments=${materialState.attachmentCount}`);
  }

  // Strategy 5: keyboard.insertText/type only for small prompts.  For your
  // 600k+ prompt this path is intentionally skipped so it cannot block for
  // minutes before we ever reach submit.
  let insertTextOk = false;
  let typedOk = false;
  if (!hasPromptMaterialState(materialState) && fullPrompt.length <= SMALL_PROMPT_INSERT_TEXT_MAX_CHARS) {
    try {
      await focusPromptBox(page, 5000);
      await page.keyboard.insertText(fullPrompt);
      insertTextOk = true;
    } catch (err) {
      vwarn("submitPrompt: keyboard.insertText failed", { err: err.message });
    }
    await refreshMaterial(`keyboardInsertText:${insertTextOk}`);
    if (!hasPromptMaterialState(materialState)) {
      typedOk = await typePrompt(page, fullPrompt);
      await refreshMaterial(`keyboardType:${typedOk}`);
    }
  }

  // Last chance: if an attachment chip exists and exposes Show in text field,
  // expand it before the submit phase.
  if (lenAfter < threshold) await showAttachmentIfAvailable("final-pre-submit");

  await dumpComposerState(page, "post-insert");
  materialState = await promptMaterialState(page);
  lenAfter = materialState.composerLen || 0;
  vlog("submitPrompt: prompt material after insert", materialState);

  if (!hasPromptMaterialState(materialState)) {
    throw new Error(`All paste strategies failed to populate the composer or create prompt material. strategies=${strategyResults.join(",")}`);
  }

  if (fullPrompt.length >= 10000 && lenAfter < threshold) {
    vwarn("submitPrompt: composer text is below expected threshold; proceeding because prompt material exists", {
      lenAfter,
      fullPromptLen: fullPrompt.length,
      threshold,
      attachmentCount: materialState.attachmentCount,
      showInTextFieldCount: materialState.showInTextFieldCount,
      attachmentHints: materialState.attachmentHints,
      strategyResults,
    });
  }

  vlog("submitPrompt: prompt material ready", {
    lenAfter,
    fullPromptLen: fullPrompt.length,
    threshold,
    attachmentCount: materialState.attachmentCount,
    showInTextFieldCount: materialState.showInTextFieldCount,
    insertElapsedMs: Date.now() - insertStartedAt,
    strategyResults,
  });

  const deadline = Date.now() + promptTimeoutMs;
  const startedAt = Date.now();
  let iter = 0;
  let lastShortLog = 0;
  let lastVerboseLog = 0;
  let clickAttempts = 0;
  let keyboardSendAttempted = false;
  let bareEnterAttempted = false;
  let loopShowAttempted = false;

  while (Date.now() < deadline) {
    iter++;
    const loopMaterial = await promptMaterialState(page);
    const length = loopMaterial.composerLen || 0;
    const hasMaterial = hasPromptMaterialState(loopMaterial);
    const sendInfo = await inspectSendButton(page);
    const generating = await isGenerating(page);
    const currentAssistantCount = await assistantCount(page);

    if (generating || currentAssistantCount > beforeCount) {
      vlog("submitPrompt: submission detected", {
        iter,
        generating,
        currentAssistantCount,
        beforeCount,
        clickAttempts,
        keyboardSendAttempted,
      });
      return;
    }

    if (clickAttempts > 0 && !hasMaterial) {
      vlog("submitPrompt: prompt material cleared after click -> treating as submitted", { iter, clickAttempts });
      return;
    }

    if (!loopShowAttempted && loopMaterial.showInTextFieldCount > 0 && length < threshold) {
      loopShowAttempted = true;
      vlog("submitPrompt: attachment still has Show in text field; trying before send", {
        iter,
        length,
        attachmentCount: loopMaterial.attachmentCount,
        showInTextFieldCount: loopMaterial.showInTextFieldCount,
        attachmentHints: loopMaterial.attachmentHints,
      });
      await tryShowAttachmentInTextField(page);
      await page.waitForTimeout(700);
      continue;
    }

    if (sendInfo && hasMaterial && !sendInfo.disabled) {
      clickAttempts++;
      vlog("submitPrompt: enabled send button found -> clicking", {
        iter,
        clickAttempts,
        length,
        attachmentCount: loopMaterial.attachmentCount,
        showInTextFieldCount: loopMaterial.showInTextFieldCount,
        sendSelector: sendInfo.selector,
        aria: sendInfo.aria,
        testid: sendInfo.testid,
      });
      const clickResult = await clickSendButton(page);
      vlog("submitPrompt: clickSendButton result", clickResult);
      await page.waitForTimeout(650);
      continue;
    }

    const elapsedMs = Date.now() - startedAt;
    if (!keyboardSendAttempted && hasMaterial && elapsedMs > 2500) {
      vlog("submitPrompt: send button not enabled quickly; trying Cmd/Ctrl+Enter fallback");
      keyboardSendAttempted = await sendViaKeyboard(page, false);
      await page.waitForTimeout(800);
      continue;
    }

    if (!bareEnterAttempted && keyboardSendAttempted && hasMaterial && elapsedMs > 9000 && (!sendInfo || sendInfo.disabled)) {
      vlog("submitPrompt: trying bare Enter fallback after Cmd/Ctrl+Enter did not submit");
      bareEnterAttempted = await sendViaKeyboard(page, true);
      await page.waitForTimeout(800);
      continue;
    }

    if (Date.now() - lastShortLog > 2000) {
      vlog("submitPrompt: waiting for submit to take", {
        iter,
        elapsedMs,
        composerLen: length,
        hasMaterial,
        attachmentCount: loopMaterial.attachmentCount,
        showInTextFieldCount: loopMaterial.showInTextFieldCount,
        sendVisible: !!sendInfo,
        sendDisabled: sendInfo ? sendInfo.disabled : null,
        sendAria: sendInfo ? sendInfo.aria : null,
        sendTestid: sendInfo ? sendInfo.testid : null,
        generating,
        clickAttempts,
        keyboardSendAttempted,
        bareEnterAttempted,
      });
      lastShortLog = Date.now();
    }

    if (Date.now() - lastVerboseLog > 20000) {
      await dumpComposerState(page, `submit-loop-iter-${iter}`);
      lastVerboseLog = Date.now();
    }

    await page.waitForTimeout(250);
  }

  const timeoutMaterial = await promptMaterialState(page);
  const length = timeoutMaterial.composerLen || 0;
  const sendInfo = await inspectSendButton(page);
  await dumpComposerState(page, "submit-timeout");
  throw new Error(
    `Prompt was not submitted before timeout; composerLen=${length}, ` +
      `attachmentCount=${timeoutMaterial.attachmentCount}, showInTextFieldCount=${timeoutMaterial.showInTextFieldCount}, ` +
      `sendVisible=${!!sendInfo}, sendDisabled=${sendInfo ? sendInfo.disabled : "n/a"}, ` +
      `sendAria=${sendInfo ? JSON.stringify(sendInfo.aria) : "n/a"}, ` +
      `clickAttempts=${clickAttempts}, keyboardSendAttempted=${keyboardSendAttempted}, ` +
      `bareEnterAttempted=${bareEnterAttempted}, strategies=${strategyResults.join(",")}`
  );
}

async function waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs, downloadContract = null) {
  vlog("waitForNewAssistantToSettle: start", {
    beforeCount,
    responseTimeoutMs,
    hasDownloadContract: !!(downloadContract && !downloadContract.error),
    bundleZip: downloadContract && downloadContract.bundleZip ? downloadContract.bundleZip : null,
    applyScript: downloadContract && downloadContract.applyScript ? downloadContract.applyScript : null,
  });
  let previous = "";
  let stableCount = 0;
  let lastNonEmpty = "";
  let sawNewMessage = false;
  let lastShortLog = 0;
  let iter = 0;
  const deadline = Date.now() + responseTimeoutMs;

  while (Date.now() < deadline) {
    iter++;
    await page.waitForTimeout(1000);
    const loc = page.locator('[data-message-author-role="assistant"]');
    const count = await loc.count().catch(() => 0);
    let current = "";
    if (count > beforeCount) {
      sawNewMessage = true;
      current = await loc.nth(count - 1).innerText().catch(() => "");
    } else {
      current = await lastAssistantText(page);
    }
    current = current.trim();
    if (current) lastNonEmpty = current;

    const generating = await isGenerating(page);

    if (Date.now() - lastShortLog > 10000) {
      const elapsedMs = responseTimeoutMs - (deadline - Date.now());
      vlog("settle progress", {
        iter,
        elapsedMs,
        assistantCount: count,
        sawNew: sawNewMessage,
        generating,
        currentLen: current.length,
        currentPreview: current.slice(0, 80),
        stable: stableCount,
      });
      lastShortLog = Date.now();
    }

    if (downloadContract && !downloadContract.error && !generating) {
      const artifactLinksReady = await visibleDownloadBundleReady(page, downloadContract);
      if (artifactLinksReady) {
        vlog("settle: exact current-contract downloadable bundle links are visible -> done", {
          iter,
          bundleZip: downloadContract.bundleZip,
          applyScript: downloadContract.applyScript,
          finalLen: current.length,
        });
        await waitForComposerReadyForNextPrompt(page, 120000, "follow-up after visible bundle links");
        return { text: current || lastNonEmpty, settled: true, sawNewMessage, artifactLinksReady: true };
      }
    }

    if (current && current === previous && !generating && sawNewMessage) {
      stableCount++;
      if (stableCount >= 5) {
        vlog("settle: stable for 5 iters -> done", { iter, finalLen: current.length });
        await waitForComposerReadyForNextPrompt(page, 120000, "follow-up after response");
        return { text: current, settled: true, sawNewMessage };
      }
    } else {
      if (stableCount > 0) {
        vlog("settle: stability streak broken", {
          iter,
          textChanged: current !== previous,
          generating,
          sawNew: sawNewMessage,
        });
      }
      stableCount = 0;
      previous = current;
    }
  }
  vwarn("settle: response did not settle within timeout", {
    responseTimeoutMs,
    sawNewMessage,
    lastLen: lastNonEmpty.length,
  });
  await waitForComposerReadyForNextPrompt(page, 30000, "follow-up after response timeout");
  return { text: lastNonEmpty, settled: false, sawNewMessage };
}

// ---------------------------------------------------------------------------
// Browser / tab handling
// ---------------------------------------------------------------------------
async function denyMicIfPossible(browser) {
  try {
    const session = await browser.newBrowserCDPSession();
    for (const permissionName of ["audioCapture", "microphone"]) {
      try {
        await session.send("Browser.setPermission", {
          permission: { name: permissionName },
          setting: "denied",
          origin: "https://chatgpt.com",
        });
      } catch (_) {}
    }
  } catch (_) {}
}

async function dumpDebugState(page, metaDir) {
  await page.screenshot({ path: path.join(metaDir, "after-generation.png"), fullPage: false }).catch(() => {});
  const candidates = await page
    .evaluate(() => {
      const els = [...document.querySelectorAll("a, button")];
      return els.map((el, i) => {
        const rect = el.getBoundingClientRect();
        return {
          i,
          tag: el.tagName.toLowerCase(),
          text: (el.innerText || el.textContent || "").trim().slice(0, 250),
          aria: el.getAttribute("aria-label") || "",
          title: el.getAttribute("title") || "",
          href: el.getAttribute("href") || "",
          download: el.getAttribute("download") || "",
          testid: el.getAttribute("data-testid") || "",
          className: String(el.getAttribute("class") || "").slice(0, 250),
          visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
          rect: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          },
        };
      });
    })
    .catch(() => []);
  fs.writeFileSync(path.join(metaDir, "click-candidates.json"), JSON.stringify(candidates, null, 2), "utf8");
  vlog("dumpDebugState: wrote screenshot + click-candidates.json", { metaDir });
}

async function connectBrowser(connectTimeoutMs) {
  return await chromium.connectOverCDP(CDP_URL, { timeout: connectTimeoutMs });
}

async function newChatPage(browser, pageTimeoutMs) {
  vlog("newChatPage: opening new tab");
  const context = browser.contexts()[0];
  if (!context) throw new Error(`No existing Chrome context found at ${CDP_URL}`);
  context.setDefaultTimeout(Math.min(pageTimeoutMs, 30000));
  const page = await context.newPage();
  await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
  await grantClipboardPermissions(page).catch(() => {});
  vlog("newChatPage: navigating to chatgpt.com");
  await page.goto("https://chatgpt.com/", { waitUntil: "domcontentloaded", timeout: pageTimeoutMs });
  // No fixed hydration sleep.  Wait directly for the composer with the fast
  // parallel selector finder.
  await findPromptBox(page, Math.min(pageTimeoutMs, 30000));
  vlog("newChatPage: prompt box visible", { url: page.url() });
  return page;
}

async function currentExistingChatPage(browser, pageTimeoutMs, composerWaitMs) {
  const candidates = await listChatGptPages(browser);
  const targetUrl = process.env.GPT_WEB_CURRENT_CHAT_URL || "";
  const targetConv = conversationIdFromUrl(targetUrl);
  const conversationCandidates = candidates.filter((c) => isConversationUrl(c.url));
  vlog("currentExistingChatPage: candidate ChatGPT tabs", {
    count: candidates.length,
    conversationCount: conversationCandidates.length,
    targetUrl,
    urls: candidates.map((c) => c.url),
  });

  if (conversationCandidates.length === 0) {
    throw new Error(
      "current_tab mode requires an already-open ChatGPT conversation tab (/c/<chat-id>). " +
        "Open the existing chat in Chrome, then rerun; or pass --new if you intentionally want a fresh chat."
    );
  }

  const ordered = conversationCandidates.slice().reverse().sort((a, b) => {
    if (!targetConv) return 0;
    const aMatch = conversationIdFromUrl(a.url) === targetConv ? 1 : 0;
    const bMatch = conversationIdFromUrl(b.url) === targetConv ? 1 : 0;
    return bMatch - aMatch;
  });

  const failures = [];
  for (const candidate of ordered) {
    const page = candidate.page;
    if (page.isClosed()) continue;
    const originalUrl = page.url();
    try {
      await page.bringToFront();
      await page.waitForLoadState("domcontentloaded", { timeout: 3000 }).catch(() => {});
      await grantClipboardPermissions(page).catch(() => {});
      if (await waitForComposerReadyForNextPrompt(page, Math.min(composerWaitMs, 30000), "existing-chat reuse")) {
        vlog("currentExistingChatPage: reusing existing conversation tab", { url: page.url(), title: candidate.title });
        return page;
      }

      // If the composer is stale/hidden in an existing conversation, reload the
      // same conversation URL.  Do not open a fresh tab or navigate to /.
      if (isConversationUrl(originalUrl)) {
        vlog("currentExistingChatPage: composer not ready; reloading same existing conversation", { url: originalUrl });
        await page.goto(originalUrl, { waitUntil: "domcontentloaded", timeout: pageTimeoutMs }).catch((err) => {
          failures.push(`${originalUrl}: reload failed: ${err.message}`);
        });
        await page.waitForTimeout(2500);
        await grantClipboardPermissions(page).catch(() => {});
        if (await waitForComposerReadyForNextPrompt(page, Math.min(composerWaitMs, 30000), "existing-chat after reload")) {
          vlog("currentExistingChatPage: reusing existing conversation after reload", { url: page.url(), title: candidate.title });
          return page;
        }
      }
      failures.push(`${originalUrl}: composer not ready`);
    } catch (err) {
      failures.push(`${originalUrl}: ${err.message}`);
      vwarn("currentExistingChatPage: candidate threw", { url: originalUrl, err: err.message });
    }
  }

  throw new Error(
    "Found ChatGPT conversation tab(s), but none had a ready composer. " +
      "The driver will not open a fresh chat in current_tab mode. " +
      `Candidates: ${failures.join(" | ")}`
  );
}

async function findTabByUrl(browser, targetUrl) {
  if (!targetUrl) return null;
  const targetConv = conversationIdFromUrl(targetUrl);
  const pages = await listChatGptPages(browser);
  for (const candidate of pages) {
    if (candidate.page.isClosed()) continue;
    if (candidate.url === targetUrl) return candidate.page;
  }
  if (targetConv) {
    for (const candidate of pages) {
      if (candidate.page.isClosed()) continue;
      if (conversationIdFromUrl(candidate.url) === targetConv) return candidate.page;
    }
  }
  if (!isConversationUrl(targetUrl)) {
    const open = pages.filter((c) => !c.page.isClosed());
    if (open.length === 1) {
      vlog("findTabByUrl: matched the only open ChatGPT tab", {
        markerUrl: targetUrl,
        tabUrl: open[0].url,
      });
      return open[0].page;
    }
  }
  return null;
}


// ---------------------------------------------------------------------------
// Download-bundle artifact transport
// ---------------------------------------------------------------------------
const DOWNLOAD_TRANSPORT = "vc4_codegen_download_bundle_v1";

function parseDownloadBundleContract(promptText) {
  const text = String(promptText || "");
  if (!text.includes(DOWNLOAD_TRANSPORT)) return null;

  function contractFromObject(obj) {
    if (!obj || typeof obj !== "object") return null;
    if (obj.transport !== DOWNLOAD_TRANSPORT) return null;
    const artifactPrefix = typeof obj.artifact_prefix === "string" ? obj.artifact_prefix : "";
    const bundleZip = typeof obj.bundle_zip === "string"
      ? obj.bundle_zip
      : (typeof obj.bundle_zip_filename === "string" ? obj.bundle_zip_filename : "");
    const applyScript = typeof obj.apply_script === "string"
      ? obj.apply_script
      : (typeof obj.apply_script_filename === "string" ? obj.apply_script_filename : "");
    if (!bundleZip || !applyScript) return null;
    return { transport: DOWNLOAD_TRANSPORT, artifactPrefix, bundleZip, applyScript };
  }

  // Prefer the current prompt's explicit DOWNLOAD_CONTRACT_JSON block. Failure
  // packets and chat history can contain stale artifact filenames from earlier
  // attempts; those must never override the current contract.
  const jsonBlocks = [];
  const fenceRe = /`{3,4}json\s*([\s\S]*?)`{3,4}/g;
  let match;
  while ((match = fenceRe.exec(text)) !== null) {
    jsonBlocks.push({ index: match.index, body: match[1] });
  }
  for (let i = jsonBlocks.length - 1; i >= 0; --i) {
    try {
      const parsed = JSON.parse(jsonBlocks[i].body);
      const contract = contractFromObject(parsed);
      if (contract) return validateDownloadBundleContract(contract);
    } catch (_) {
      // Ignore unrelated/non-JSON context blocks.
    }
  }

  // Fallback for older prompts: use the last occurrence, not the first. The
  // renderer appends the current contract after failure/context material.
  function lastString(keys, suffixRe) {
    for (const key of keys) {
      const re = new RegExp(`"${escapeRegExp(key)}"\\s*:\\s*"([^"]+)"`, "gm");
      let found = "";
      let m;
      while ((m = re.exec(text)) !== null) {
        if (!suffixRe || suffixRe.test(m[1])) found = m[1];
      }
      if (found) return found;
    }
    return "";
  }

  let artifactPrefix = lastString(["artifact_prefix"], null);
  let bundleZip = lastString(["bundle_zip", "bundle_zip_filename"], /\.zip$/);
  let applyScript = lastString(["apply_script", "apply_script_filename"], /\.sh$/);

  if (!artifactPrefix) {
    const re = /ARTIFACT_PREFIX\s*[:=]\s*([A-Za-z0-9_.-]+)/g;
    let m;
    while ((m = re.exec(text)) !== null) artifactPrefix = m[1];
  }
  if (!bundleZip && artifactPrefix) bundleZip = `${artifactPrefix}.zip`;
  if (!applyScript && artifactPrefix) applyScript = `${artifactPrefix}.sh`;

  return validateDownloadBundleContract({ transport: DOWNLOAD_TRANSPORT, artifactPrefix, bundleZip, applyScript });
}

function validateDownloadBundleContract(contract) {
  const bundleZip = contract.bundleZip || "";
  const applyScript = contract.applyScript || "";
  if (!bundleZip || !applyScript) {
    return {
      transport: DOWNLOAD_TRANSPORT,
      error: "prompt mentions downloadable bundle transport but does not contain bundle_zip/apply_script filenames",
      artifactPrefix: contract.artifactPrefix || "",
      bundleZip,
      applyScript,
    };
  }
  for (const name of [bundleZip, applyScript]) {
    if (name !== path.basename(name) || name.includes("..") || name.includes("/") || name.includes("\\")) {
      return { transport: DOWNLOAD_TRANSPORT, error: `unsafe artifact filename in prompt: ${name}` };
    }
  }
  return {
    transport: DOWNLOAD_TRANSPORT,
    artifactPrefix: contract.artifactPrefix || "",
    bundleZip,
    applyScript,
  };
}

function hasExactDownloadBundleLinks(candidates, contract) {
  if (!contract || contract.error) return false;
  const wanted = [contract.bundleZip, contract.applyScript].filter(Boolean);
  if (wanted.length !== 2) return false;
  const seen = new Set();
  for (const c of candidates || []) {
    if (!c || c.visible === false) continue;
    const fields = [c.text, c.aria, c.title, c.download, c.href].map((x) => String(x || ""));
    for (const filename of wanted) {
      if (fields.some((field) => field.includes(filename))) seen.add(filename);
    }
  }
  return wanted.every((filename) => seen.has(filename));
}

async function visibleDownloadBundleReady(page, contract) {
  if (!contract || contract.error) return false;
  const candidates = await page
    .evaluate(() => {
      const els = [...document.querySelectorAll("a, button")];
      return els.map((el) => ({
        tag: el.tagName.toLowerCase(),
        text: (el.innerText || el.textContent || "").trim(),
        aria: el.getAttribute("aria-label") || "",
        title: el.getAttribute("title") || "",
        href: el.getAttribute("href") || "",
        download: el.getAttribute("download") || "",
        visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
      }));
    })
    .catch(() => []);
  return hasExactDownloadBundleLinks(candidates, contract);
}

function statStableEnough(filePath, minMtimeMs) {
  try {
    const st1 = fs.statSync(filePath);
    if (!st1.isFile()) return null;
    if (st1.mtimeMs < minMtimeMs) return null;
    const size1 = st1.size;
    const st2 = fs.statSync(filePath);
    if (st2.size !== size1) return null;
    return { path: filePath, bytes: st2.size, mtimeMs: st2.mtimeMs };
  } catch (_) {
    return null;
  }
}

function copyIfRecentDownload(filename, dest, minMtimeMs) {
  const downloadsDir = path.join(os.homedir(), "Downloads");
  const exact = path.join(downloadsDir, filename);
  const exactStat = statStableEnough(exact, minMtimeMs);
  if (exactStat) {
    mkdirp(path.dirname(dest));
    fs.copyFileSync(exact, dest);
    return { ok: true, method: "downloads-folder-exact", source: exact, dest, ...exactStat };
  }
  return { ok: false, method: "downloads-folder-exact", filename, lookedIn: downloadsDir };
}

async function waitForDownloadFile(filename, dest, minMtimeMs, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastLog = 0;
  while (Date.now() < deadline) {
    const copied = copyIfRecentDownload(filename, dest, minMtimeMs);
    if (copied.ok) return copied;
    if (Date.now() - lastLog > 5000) {
      vlog("artifact download poll: waiting for file in ~/Downloads", {
        filename,
        remainingMs: Math.max(0, deadline - Date.now()),
      });
      lastLog = Date.now();
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return { ok: false, method: "downloads-folder-timeout", filename, timeoutMs };
}

async function findAndClickDownloadLink(page, filename) {
  return await page
    .evaluate((filename) => {
      function visible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }
      function labelParts(el) {
        const href = el.getAttribute("href") || "";
        const text = ((el.innerText || el.textContent || "") + "").trim();
        return [
          el.getAttribute("download") || "",
          el.getAttribute("aria-label") || "",
          el.getAttribute("title") || "",
          href,
          (() => {
            try {
              const u = new URL(href, location.href);
              return decodeURIComponent((u.pathname || "").split("/").pop() || "");
            } catch (_) {
              return "";
            }
          })(),
          text,
        ].map((s) => String(s || "").replace(/\s+/g, " ").trim()).filter(Boolean);
      }
      function candidateBlob(el) {
        return labelParts(el).join(" ").replace(/\s+/g, " ").trim();
      }
      function filenameMatches(el) {
        const parts = labelParts(el);
        return parts.some((part) => part === filename || part.includes(filename));
      }
      function scan(root, scope) {
        const candidates = Array.from(root.querySelectorAll('a, button, [role="button"]'));
        for (const el of candidates) {
          if (!visible(el)) continue;
          if (!filenameMatches(el)) continue;
          el.scrollIntoView({ block: "center", inline: "center" });
          el.click();
          return {
            ok: true,
            scope,
            tag: el.tagName.toLowerCase(),
            label: candidateBlob(el).slice(0, 300),
          };
        }
        return null;
      }

      // Prefer the newest assistant turn.  Exact filenames are unique per
      // attempt, but scoping avoids stale controls in long accumulated chats.
      const assistantTurns = Array.from(document.querySelectorAll('[data-message-author-role="assistant"]'));
      if (assistantTurns.length) {
        for (let i = assistantTurns.length - 1; i >= 0; --i) {
          const found = scan(assistantTurns[i], `assistant-turn-${i}`);
          if (found) return found;
          // Only scan older assistant turns when their text mentions the exact
          // filename; this keeps retries from clicking a generic stale download.
          const text = ((assistantTurns[i].innerText || assistantTurns[i].textContent || "") + "");
          if (!text.includes(filename)) break;
        }
      }

      const globalFound = scan(document.body, "document");
      if (globalFound) return globalFound;

      const visibleLabels = Array.from(document.querySelectorAll('a, button, [role="button"]'))
        .filter(visible)
        .map((el) => candidateBlob(el).slice(0, 160))
        .filter(Boolean)
        .slice(-40);
      return {
        ok: false,
        reason: `no visible link/button containing exact filename ${filename}`,
        visibleLabels,
      };
    }, filename)
    .catch((err) => ({ ok: false, reason: String(err) }));
}

async function clickDownloadAndSave(page, filename, dest, timeoutMs) {
  mkdirp(path.dirname(dest));
  const downloadPromise = page.waitForEvent("download", { timeout: timeoutMs }).catch((err) => ({ __downloadError: err }));
  const clickResult = await findAndClickDownloadLink(page, filename);
  if (!clickResult.ok) {
    return { ok: false, method: "click-download", filename, clickResult };
  }
  const download = await downloadPromise;
  if (download && download.__downloadError) {
    return {
      ok: false,
      method: "click-download",
      filename,
      clickResult,
      error: download.__downloadError.message || String(download.__downloadError),
    };
  }
  try {
    await download.saveAs(dest);
    const st = fs.statSync(dest);
    return {
      ok: true,
      method: "click-download",
      filename,
      dest,
      bytes: st.size,
      suggestedFilename: download.suggestedFilename ? download.suggestedFilename() : "",
      clickResult,
    };
  } catch (err) {
    return { ok: false, method: "click-download", filename, clickResult, error: err.message || String(err) };
  }
}

async function collectDownloadBundleArtifacts({ page, contract, outDir, metaDir, sinceMs, timeoutMs }) {
  const downloadDir = path.join(outDir, "downloads");
  mkdirp(downloadDir);
  const minMtimeMs = Math.max(0, sinceMs - 1000);
  const wanted = [
    { role: "bundle_zip", filename: contract.bundleZip, canonical: "bundle.zip" },
    { role: "apply_script", filename: contract.applyScript, canonical: "apply_bundle.sh" },
  ];
  const files = {};
  const attempts = [];

  for (const item of wanted) {
    const dest = path.join(downloadDir, item.filename);
    const canonicalDest = path.join(outDir, item.canonical);
    let result = await clickDownloadAndSave(page, item.filename, dest, Math.min(timeoutMs, 45000));
    attempts.push({ role: item.role, filename: item.filename, phase: "click", result });
    if (!result.ok) {
      result = await waitForDownloadFile(item.filename, dest, minMtimeMs, timeoutMs);
      attempts.push({ role: item.role, filename: item.filename, phase: "poll-downloads", result });
    }
    if (!result.ok) {
      return {
        ok: false,
        error: `missing downloadable artifact ${item.filename}`,
        contract,
        attempts,
        downloaded_files: files,
      };
    }
    fs.copyFileSync(dest, canonicalDest);
    if (item.canonical.endsWith(".sh")) {
      try { fs.chmodSync(canonicalDest, 0o755); } catch (_) {}
      try { fs.chmodSync(dest, 0o755); } catch (_) {}
    }
    files[item.role] = path.relative(outDir, dest).replace(/\\/g, "/");
    files[`${item.role}_canonical`] = item.canonical;
  }

  const transport = {
    schema_version: 1,
    transport: DOWNLOAD_TRANSPORT,
    artifact_prefix: contract.artifactPrefix,
    bundle_zip: contract.bundleZip,
    apply_script: contract.applyScript,
    downloaded_files: files,
    attempts,
    downloads_dir: path.join(os.homedir(), "Downloads"),
    collected_at: new Date().toISOString(),
  };
  fs.writeFileSync(path.join(outDir, "artifact_transport.json"), JSON.stringify(transport, null, 2) + "\n", "utf8");
  fs.writeFileSync(path.join(metaDir, "artifact_transport.json"), JSON.stringify(transport, null, 2) + "\n", "utf8");
  return { ok: true, ...transport };
}

// ---------------------------------------------------------------------------
// Prompt wrapper
// ---------------------------------------------------------------------------
function buildWrappedPrompt(userPrompt, token) {
  const wantsDownloadBundle = String(userPrompt || "").includes(DOWNLOAD_TRANSPORT);
  if (wantsDownloadBundle) {
    const downloadContract = parseDownloadBundleContract(userPrompt) || {};
    const bundleZip = downloadContract.bundleZip || "<exact bundle zip filename from the prompt>";
    const applyScript = downloadContract.applyScript || "<exact apply shell filename from the prompt>";
    return `
You are generating downloadable files for local automation.

The user prompt below contains a VC4 downloadable bundle contract (${DOWNLOAD_TRANSPORT}). Follow that contract exactly:
- Create the required downloadable zip and shell script with the exact filenames named in the prompt.
- The visible download link or attachment label for the zip MUST be exactly: ${bundleZip}
- The visible download link or attachment label for the shell script MUST be exactly: ${applyScript}
- Do not use generic link labels such as "Download bundle zip" or "Download apply script".
- Repeat the same exact filenames in the final status JSON fields "bundle_zip" and "apply_script".
- Do not paste source files, patches, or large code blocks into the chat response.
- Do not use GPTWEB_FILE blocks for implementation patches when the downloadable bundle contract is present.
- A short status JSON in the chat is fine, but the downloadable files are the source of truth.

The final chat response should contain only a compact status JSON plus the two downloadable file links/attachments labeled with the exact filenames above.

User prompt:
${userPrompt}
`.trim();
  }
  return `
You are generating files for local automation.

Generate the requested file or files as plain text using the exact block format below. My local automation will parse these blocks and write them to disk.

Required file-block format:

BEGIN_GPTWEB_FILE token=${token} path=<relative/path>
<complete file contents>
END_GPTWEB_FILE token=${token}

Rules:
- Output only GPTWEB_FILE blocks.
- You may output one file block or many file blocks.
- Every file must have its own block.
- Paths must be relative.
- Do not use absolute paths.
- Do not use ".." in paths.
- Do not write into .gpt-web-run.
- Do not wrap the file blocks in Markdown fences.
- Do not add explanations outside the file blocks.
- Preserve the exact requested source/code/markdown content inside each block.

User prompt:
${userPrompt}
`.trim();
}

function resolveRepoRoot(args, outDir) {
  if (args["repo-root"]) {
    return path.resolve(args["repo-root"]);
  }
  let candidate = path.resolve(outDir);
  for (let i = 0; i < 10; i++) {
    if (fs.existsSync(path.join(candidate, ".vc4_auto"))) return candidate;
    if (fs.existsSync(path.join(candidate, ".git"))) return candidate;
    const parent = path.dirname(candidate);
    if (parent === candidate) break;
    candidate = parent;
  }
  return process.cwd();
}

// ---------------------------------------------------------------------------
// Main run
// ---------------------------------------------------------------------------
async function run(mode, argv) {
  const args = parseArgs(argv);
  const promptFile = requireArg(args, "prompt-file");
  const outDir = path.resolve(requireArg(args, "out"));
  const metaDir = path.join(outDir, ".gpt-web-run");
  const closeTab = boolArg(args, "close-tab");
  const connectTimeoutMs = intArg(args, "connect-timeout-ms", 120000);
  const pageTimeoutMs = intArg(args, "page-timeout-ms", 120000);
  const promptTimeoutMs = intArg(args, "prompt-timeout-ms", 120000);
  const responseTimeoutMs = intArg(args, "response-timeout-ms", 30 * 60 * 1000);
  const composerWaitMs = intArg(args, "composer-wait-ms", 5 * 60 * 1000);
  const allowDuplicateSubmit = boolArg(args, "allow-duplicate-submit");
  const artifactDownloadTimeoutMs = intArg(args, "artifact-download-timeout-ms", 3 * 60 * 1000);

  mkdirp(outDir);
  mkdirp(metaDir);
  setDebugLog(path.join(metaDir, "debug.log"));

  vlog("driver start", {
    mode,
    promptFile,
    outDir,
    metaDir,
    cdp: CDP_URL,
    pid: process.pid,
    platform: process.platform,
    nodeVersion: process.version,
  });
  vlog("driver timeouts", {
    connectTimeoutMs,
    pageTimeoutMs,
    promptTimeoutMs,
    responseTimeoutMs,
    composerWaitMs,
    artifactDownloadTimeoutMs,
  });

  const repoRoot = resolveRepoRoot(args, outDir);
  vlog("repo root resolved", { repoRoot, marker: markerPath(repoRoot) });

  const userPrompt = readPromptFile(promptFile);
  const promptHash = sha256(userPrompt);
  vlog("prompt loaded", {
    file: promptFile,
    chars: userPrompt.length,
    hash: promptHash.slice(0, 16),
    safeTransportLimitChars: promptCharLimit(),
  });
  enforcePromptCharLimit(promptFile, userPrompt.length, outDir);
  const promptDownloadContract = parseDownloadBundleContract(userPrompt);
  vlog("download contract parsed", {
    hasContract: !!promptDownloadContract,
    error: promptDownloadContract && promptDownloadContract.error ? promptDownloadContract.error : null,
    bundleZip: promptDownloadContract && promptDownloadContract.bundleZip ? promptDownloadContract.bundleZip : null,
    applyScript: promptDownloadContract && promptDownloadContract.applyScript ? promptDownloadContract.applyScript : null,
  });

  vlog("connecting to Chrome via CDP...");
  const browser = await connectBrowser(connectTimeoutMs);
  vlog("connected to Chrome");
  let page = null;
  let resumed = false;
  let token = null;
  let fullPrompt = null;

  try {
    await denyMicIfPossible(browser);

    const marker = readMarker(repoRoot);
    vlog("marker check", {
      hasMarker: !!marker,
      markerHash: marker ? String(marker.promptHash || "").slice(0, 16) : null,
      myHash: promptHash.slice(0, 16),
      hashMatches: !!(marker && marker.promptHash === promptHash),
    });

    if (marker && marker.promptHash === promptHash && marker.tabUrl && marker.token) {
      const ageMs = Date.now() - Number(marker.submittedAt || 0);
      vlog("resume candidate marker", { tabUrl: marker.tabUrl, ageMs });
      const existing = await findTabByUrl(browser, marker.tabUrl);
      if (existing) {
        vlog("RESUMING: reusing existing tab without resubmitting", { url: existing.url() });
        page = existing;
        token = marker.token;
        resumed = true;
        await page.bringToFront().catch(() => {});
      } else {
        vwarn("marker present but matching tab is gone; discarding marker");
        clearMarker(repoRoot);
      }
    } else if (marker && marker.promptHash !== promptHash) {
      vlog(
        "marker exists for a different prompt hash; will replace it after submit if no other generation is active"
      );
    }

    if (!resumed) {
      vlog("opening tab", { mode });
      page =
        mode === "new"
          ? await newChatPage(browser, pageTimeoutMs)
          : await currentExistingChatPage(browser, pageTimeoutMs, composerWaitMs);
      vlog("tab ready", { url: page.url() });

      const context = page.context();
      context.setDefaultTimeout(Math.min(pageTimeoutMs, 30000));
      await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
      await page.bringToFront();

      const existingMarker = readMarker(repoRoot);
      if (existingMarker && existingMarker.promptHash !== promptHash && !allowDuplicateSubmit) {
        const existingPage = await findTabByUrl(browser, existingMarker.tabUrl);
        if (existingPage && (await isGenerating(existingPage).catch(() => false))) {
          throw new Error(
            "Refusing to submit: a different in-flight prompt is still generating in another " +
              `ChatGPT tab. Wait for it to finish, clear ${markerPath(repoRoot)}, or pass ` +
              "--allow-duplicate-submit."
          );
        }
      }

      try {
        await findPromptBox(page, 10000);
      } catch (err) {
        throw new Error(`Prompt box not visible before submit: ${err.message}`);
      }

      token = randomToken();
      fullPrompt = buildWrappedPrompt(userPrompt, token);
      vlog("wrapped prompt built", { token, len: fullPrompt.length });

      const beforeCount = await assistantCount(page);
      vlog("pre-submit assistant count", { beforeCount });

      await dumpComposerState(page, "pre-submit");
      await submitPrompt(page, fullPrompt, promptTimeoutMs);

      const tabUrl = await waitForConversationUrl(page, 15000);
      writeMarker(repoRoot, {
        promptHash,
        token,
        tabUrl,
        outDir,
        mode,
        beforeCount,
        submittedAt: Date.now(),
      });
      vlog("post-submit: waiting for response to settle", { tabUrl });

      const result = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs, promptDownloadContract);
      vlog("settle result", {
        settled: result.settled,
        sawNewMessage: result.sawNewMessage,
        textLen: (result.text || "").length,
      });
      await finalizeAnswer({
        page,
        repoRoot,
        outDir,
        metaDir,
        token,
        answer: result.text,
        settled: result.settled,
        promptFile,
        mode,
        closeTab,
        pageTimeoutMs,
        artifactDownloadTimeoutMs,
        submittedAtMs: Number((readMarker(repoRoot) || {}).submittedAt || Date.now()),
      });
      return;
    }

    let beforeCount = Number(marker.beforeCount);
    if (!Number.isFinite(beforeCount) || beforeCount < 0) {
      const cur = await assistantCount(page);
      beforeCount = Math.max(0, cur - 1);
    }
    vlog("resume: waiting for assistant message to settle", { beforeCount });
    const result = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs, promptDownloadContract);
    vlog("resume settle result", {
      settled: result.settled,
      textLen: (result.text || "").length,
    });
    await finalizeAnswer({
      page,
      repoRoot,
      outDir,
      metaDir,
      token,
      answer: result.text,
      settled: result.settled,
      promptFile,
      mode,
      closeTab,
      pageTimeoutMs,
      artifactDownloadTimeoutMs,
      submittedAtMs: Number((readMarker(repoRoot) || {}).submittedAt || Date.now()),
    });
  } finally {
    if (closeTab && page) await page.close().catch(() => {});
    if (browser && typeof browser.disconnect === "function") await browser.disconnect().catch(() => {});
    setTimeout(() => process.exit(process.exitCode || 0), 50);
  }
}

async function finalizeAnswer({
  page,
  repoRoot,
  outDir,
  metaDir,
  token,
  answer,
  settled,
  promptFile,
  mode,
  closeTab,
  pageTimeoutMs,
  artifactDownloadTimeoutMs,
  submittedAtMs,
}) {
  const answerPath = path.join(metaDir, "answer.md");
  fs.writeFileSync(answerPath, answer || "", "utf8");
  vlog("finalizeAnswer: wrote answer.md", { path: answerPath, len: (answer || "").length });

  const originalPromptText = readPromptFile(promptFile);
  const downloadContract = parseDownloadBundleContract(originalPromptText);
  let artifactResult = null;
  if (downloadContract) {
    if (downloadContract.error) {
      artifactResult = { ok: false, error: downloadContract.error, contract: downloadContract };
      fs.writeFileSync(path.join(metaDir, "artifact_transport_error.json"), JSON.stringify(artifactResult, null, 2) + "\n", "utf8");
      await dumpDebugState(page, metaDir);
      process.exitCode = 4;
    } else {
      vlog("finalizeAnswer: collecting downloadable bundle artifacts", downloadContract);
      artifactResult = await collectDownloadBundleArtifacts({
        page,
        contract: downloadContract,
        outDir,
        metaDir,
        sinceMs: Number(submittedAtMs || Date.now()),
        timeoutMs: artifactDownloadTimeoutMs,
      });
      vlog("finalizeAnswer: artifact collection result", artifactResult);
      if (!artifactResult.ok) {
        fs.writeFileSync(path.join(metaDir, "artifact_transport_error.json"), JSON.stringify(artifactResult, null, 2) + "\n", "utf8");
        await dumpDebugState(page, metaDir);
        process.exitCode = 4;
      }
    }
  }

  const parsedFiles = downloadContract ? [] : parseFileBlocks(answer || "", token);
  vlog("finalizeAnswer: parsed file blocks", { count: parsedFiles.length });
  const { written, skipped, fileMetadata } = writeGeneratedFiles(parsedFiles, outDir);
  vlog("finalizeAnswer: write summary", { written: written.length, skipped: skipped.length, fileMetadata });

  const artifactOk = !!(artifactResult && artifactResult.ok);
  const effectiveSettled = !!settled || artifactOk || written.length > 0;
  if ((!artifactOk && written.length === 0) || skipped.length > 0) {
    await dumpDebugState(page, metaDir);
  } else {
    if (
      !(await waitForComposerReadyForNextPrompt(page, 60000, "next automation step after file parse"))
    ) {
      const restoreUrl = page.url();
      if (isChatGptUrl(restoreUrl)) {
        vlog("composer still hidden; reloading current chat", { restoreUrl });
        await page
          .goto(restoreUrl, { waitUntil: "domcontentloaded", timeout: pageTimeoutMs })
          .catch(() => {});
        await page.waitForTimeout(4000);
        await waitForComposerReadyForNextPrompt(page, 60000, "next automation step after reload");
      }
    }
  }

  if (effectiveSettled) {
    clearMarker(repoRoot);
  } else {
    vlog(
      "settled=false; preserving in-flight marker so a follow-up invocation can resume waiting"
    );
  }

  const manifest = {
    ok: artifactOk || written.length > 0,
    settled: !!settled,
    effectiveSettled: !!effectiveSettled,
    mode:
      mode === "new"
        ? "new-chatgpt-tab-gptweb-file-parser"
        : "existing-chatgpt-tab-gptweb-file-parser",
    promptFile: path.resolve(promptFile),
    outDir,
    metaDir,
    answerPath,
    token,
    files: written,
    artifactResult,
    fileMetadata,
    skipped,
    parsedBlockCount: parsedFiles.length,
    keptChatGptTabOpen: !closeTab,
    chatGptTabUrl: page.url(),
    note:
      artifactOk
        ? "Downloaded bundle artifacts and staged bundle.zip/apply_bundle.sh for the local bundle applier."
        : written.length === 0
          ? "No files or downloadable bundle artifacts were written. Check .gpt-web-run/answer.md, debug.log, after-generation.png, and click-candidates.json."
          : "Parsed file blocks and wrote generated files directly under --out.",
  };
  fs.writeFileSync(path.join(metaDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  vlog("manifest written", manifest);

  if (!effectiveSettled) {
    vlog("exit code 2 (settled=false and no artifacts/files accepted)");
    process.exitCode = process.exitCode || 2;
  } else if (process.exitCode) {
    vlog("exit code preserved from artifact/file validation", { exitCode: process.exitCode });
  } else {
    vlog("exit code 0 (response accepted)", { settled, artifactOk, written: written.length });
  }
}

module.exports = { run };
