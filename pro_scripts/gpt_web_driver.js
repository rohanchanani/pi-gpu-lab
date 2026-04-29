const { chromium } = require("playwright");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const child_process = require("child_process");

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
  h = h.replace(/^path=/, "").trim();
  return normalizeRelativePath(h);
}

function parseFileBlocks(answer, expectedToken) {
  const files = [];
  const text = String(answer || "");
  const re = /^BEGIN_GPT_?WEB_FILE([^\r\n]*)\r?\n([\s\S]*?)^END_GPT_?WEB_FILE([^\r\n]*)(?:\r?\n|$)/gm;
  let m;
  while ((m = re.exec(text)) !== null) {
    const header = String(m[1] || "").trim();
    const content = m[2] ?? "";
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
    files.push({ relPath, content, header });
  }
  return files;
}

function writeGeneratedFiles(parsedFiles, outDir) {
  const written = [];
  const skipped = [];
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
      vlog("wrote file", { dest });
    } catch (err) {
      skipped.push({ relPath: f.relPath, error: err.message });
    }
  }
  return { written, skipped };
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
  '[data-testid="prompt-textarea"]',
  "#prompt-textarea",
  'div[contenteditable="true"][id="prompt-textarea"]',
  '.ProseMirror[contenteditable="true"]',
  'textarea[placeholder*="Message"]',
  "textarea",
  'div[contenteditable="true"]',
  '[contenteditable="true"]',
];

const SEND_BUTTON_SELECTORS = [
  'button[data-testid="send-button"]',
  'button[aria-label="Send prompt"]',
  'button[aria-label="Send message"]',
  'button[aria-label*="Send"]',
];

const STOP_BUTTON_SELECTORS = [
  'button[data-testid="stop-button"]',
  'button[aria-label="Stop streaming"]',
  'button[aria-label="Stop generating"]',
  'button[aria-label="Stop generating response"]',
];

async function findPromptBox(page, timeout = 10000) {
  vlog("findPromptBox: searching", { timeoutPerSelector: timeout });
  for (const selector of PROMPT_BOX_SELECTORS) {
    const loc = page.locator(selector).last();
    try {
      await loc.waitFor({ state: "visible", timeout });
      vlog("findPromptBox: matched", { selector });
      return loc;
    } catch (_) {
      vlog("findPromptBox: selector did not match", { selector });
    }
  }
  vwarn("findPromptBox: no selector matched");
  throw new Error("Could not find ChatGPT prompt box.");
}

async function composerTextLength(page) {
  return await page
    .evaluate((selectors) => {
      for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (!el) continue;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") continue;
        if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") return (el.value || "").length;
        return (el.innerText || el.textContent || "").length;
      }
      return 0;
    }, PROMPT_BOX_SELECTORS)
    .catch(() => 0);
}

async function clearPromptBoxInDom(page) {
  await page
    .evaluate((selectors) => {
      for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (!el) continue;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") continue;
        el.focus();
        if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
          el.value = "";
          el.dispatchEvent(
            new InputEvent("input", { bubbles: true, inputType: "deleteContentBackward", data: null })
          );
          return true;
        }
        const sel = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(el);
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand("delete", false, null);
        el.dispatchEvent(
          new InputEvent("input", { bubbles: true, inputType: "deleteContentBackward", data: null })
        );
        return true;
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
  for (const selector of STOP_BUTTON_SELECTORS) {
    try {
      const loc = page.locator(selector).last();
      if ((await loc.count()) > 0 && (await loc.isVisible().catch(() => false))) {
        vlog("isGenerating: stop button visible", { selector });
        return true;
      }
    } catch (_) {}
  }
  try {
    const loc = page.locator('[data-testid*="thinking" i], [data-testid*="reasoning" i]').last();
    if ((await loc.count()) > 0 && (await loc.isVisible().catch(() => false))) {
      vlog("isGenerating: thinking/reasoning pill visible");
      return true;
    }
  } catch (_) {}
  return false;
}

async function inspectSendButton(page) {
  for (const selector of SEND_BUTTON_SELECTORS) {
    const loc = page.locator(selector).last();
    const count = await loc.count().catch(() => 0);
    if (count === 0) continue;
    const visible = await loc.isVisible().catch(() => false);
    if (!visible) continue;
    let disabled = true;
    try {
      disabled = await loc.isDisabled();
    } catch (_) {}
    let aria = "";
    try {
      aria = (await loc.getAttribute("aria-label")) || "";
    } catch (_) {}
    let testid = "";
    try {
      testid = (await loc.getAttribute("data-testid")) || "";
    } catch (_) {}
    let ariaDisabled = "";
    try {
      ariaDisabled = (await loc.getAttribute("aria-disabled")) || "";
    } catch (_) {}
    return { selector, locator: loc, visible, disabled, aria, testid, ariaDisabled };
  }
  return null;
}

async function findSendButton(page) {
  const info = await inspectSendButton(page);
  return info ? info.locator : null;
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
      await findPromptBox(page, 2000);
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
    await page.waitForTimeout(1000);
  }
  vwarn("waitForComposerReadyForNextPrompt: timed out", { label, timeoutMs });
  return false;
}

// ---------------------------------------------------------------------------
// Clipboard / paste
// ---------------------------------------------------------------------------
function setClipboardText(text) {
  if (process.platform === "darwin") {
    child_process.execFileSync("/usr/bin/pbcopy", { input: text, maxBuffer: 1024 * 1024 });
    return;
  }
  if (process.env.WAYLAND_DISPLAY) {
    child_process.execFileSync("wl-copy", [], { input: text, maxBuffer: 1024 * 1024 });
    return;
  }
  child_process.execFileSync("xclip", ["-selection", "clipboard"], {
    input: text,
    maxBuffer: 1024 * 1024,
  });
}

function nativePlainTextPaste() {
  if (process.platform === "darwin") {
    const script = [
      'tell application "System Events"',
      '  keystroke "v" using {command down, shift down}',
      "end tell",
    ].join("\n");
    child_process.execFileSync("/usr/bin/osascript", ["-e", script], { stdio: "ignore" });
    return;
  }
  throw new Error("native plain paste is only implemented through osascript on macOS");
}

async function domInsertPrompt(page, fullPrompt) {
  return await page
    .evaluate(
      ({ text, selectors }) => {
        for (const selector of selectors) {
          const el = document.querySelector(selector);
          if (!el) continue;
          const style = window.getComputedStyle(el);
          if (style.visibility === "hidden" || style.display === "none") continue;
          el.focus();
          if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
            el.value = text;
            el.dispatchEvent(
              new InputEvent("input", { bubbles: true, inputType: "insertText", data: text.slice(0, 1) })
            );
            return true;
          }
          const sel = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(el);
          sel.removeAllRanges();
          sel.addRange(range);
          document.execCommand("delete", false, null);
          const ok = document.execCommand("insertText", false, text);
          el.dispatchEvent(
            new InputEvent("input", { bubbles: true, inputType: "insertText", data: text.slice(0, 1) })
          );
          return ok || (el.innerText || el.textContent || "").length > 0;
        }
        return false;
      },
      { text: fullPrompt, selectors: PROMPT_BOX_SELECTORS }
    )
    .catch(() => false);
}

// Press Cmd+Enter / Ctrl+Enter on the composer.  ChatGPT accepts this as
// "send" in many UI states, including some where the send button itself is
// in a state our locators don't recognize.  Used as a backup when
// inspectSendButton can't find a clickable send button.
async function sendViaKeyboard(page) {
  const isMac = process.platform === "darwin";
  // Focus the composer first.  Use the same selectors as findPromptBox.
  try {
    const box = await findPromptBox(page, 5000);
    await box.click({ timeout: 5000 }).catch(() => {});
  } catch (err) {
    vwarn("sendViaKeyboard: could not focus composer", { err: err.message });
    return false;
  }
  const combos = isMac ? ["Meta+Enter", "Enter"] : ["Control+Enter", "Enter"];
  for (const combo of combos) {
    try {
      vlog("sendViaKeyboard: pressing", { combo });
      await page.keyboard.press(combo);
      // Tiny pause so the page can react; the caller will detect generation
      // via assistantCount / Stop button.
      await page.waitForTimeout(500);
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
      const out = { url: location.href, candidates: [], buttons: [] };
      for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (!el) continue;
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        const visible = style.visibility !== "hidden" && style.display !== "none";
        const text =
          el.tagName === "TEXTAREA" || el.tagName === "INPUT"
            ? el.value || ""
            : el.innerText || el.textContent || "";
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
      const buttons = document.querySelectorAll(
        'button[data-testid], button[aria-label], button[type="submit"], form button'
      );
      for (const b of buttons) {
        const style = window.getComputedStyle(b);
        const visible = style.visibility !== "hidden" && style.display !== "none" && b.offsetWidth > 0;
        if (!visible) continue;
        out.buttons.push({
          testid: b.getAttribute("data-testid") || "",
          aria: b.getAttribute("aria-label") || "",
          type: b.getAttribute("type") || "",
          disabled: !!b.disabled,
          ariaDisabled: b.getAttribute("aria-disabled") || "",
          textPreview: ((b.innerText || b.textContent || "") + "").trim().slice(0, 60),
        });
      }
      return out;
    }, PROMPT_BOX_SELECTORS)
    .catch((err) => ({ error: String(err) }));
  vlog(`dumpComposerState @ ${label}`, info);
  return info;
}

// ---------------------------------------------------------------------------
// Submit core
// ---------------------------------------------------------------------------
async function submitPrompt(page, fullPrompt, promptTimeoutMs) {
  vlog("submitPrompt: start", { len: fullPrompt.length, promptTimeoutMs });
  await page.bringToFront().catch(() => {});

  vlog("submitPrompt: locating prompt box");
  const promptBox = await findPromptBox(page, Math.min(promptTimeoutMs, 60000));
  await promptBox.scrollIntoViewIfNeeded().catch(() => {});
  await promptBox.click({ timeout: 10000 }).catch((err) => {
    vwarn("submitPrompt: click on prompt box failed", { err: err.message });
  });
  vlog("submitPrompt: clearing composer");
  await clearPromptBoxInDom(page);
  await page.waitForTimeout(300);

  vlog("submitPrompt: setting clipboard");
  try {
    setClipboardText(fullPrompt);
  } catch (err) {
    vwarn("submitPrompt: clipboard set failed", { err: err.message });
  }
  vlog("submitPrompt: sending native Cmd+Shift+V paste", { len: fullPrompt.length });
  try {
    nativePlainTextPaste();
  } catch (err) {
    vwarn("submitPrompt: native paste failed", { err: err.message });
  }
  await page.waitForTimeout(500);
  await dumpComposerState(page, "post-paste");

  const deadline = Date.now() + promptTimeoutMs;
  const startedAt = Date.now();
  let lastShortLog = 0;
  let lastVerboseLog = 0;
  let ready = false;
  let iter = 0;
  let domInsertAttempted = false;
  let domInsertOk = false;
  let keyboardSendAttempted = false;
  let keyboardSendOk = false;

  // Capture initial assistantCount so the keyboard fallback can detect that
  // a send actually went through (count went up).
  const beforeKeyboardCount = await assistantCount(page);

  while (Date.now() < deadline) {
    iter++;
    const length = await composerTextLength(page);
    const sendInfo = await inspectSendButton(page);
    const generating = await isGenerating(page);

    if (sendInfo && length > 0 && !sendInfo.disabled) {
      vlog("submitPrompt: send conditions met -> clicking", {
        iter,
        length,
        sendSelector: sendInfo.selector,
        aria: sendInfo.aria,
        testid: sendInfo.testid,
      });
      try {
        await sendInfo.locator.click({ timeout: 10000 });
        ready = true;
        break;
      } catch (err) {
        vwarn("submitPrompt: send button click threw", { err: err.message });
      }
    }

    // Detect: the keyboard fallback already submitted (assistantCount went up
    // or generation started) even though we never found a clickable send btn.
    if (keyboardSendAttempted) {
      const newCount = await assistantCount(page);
      if (newCount > beforeKeyboardCount || generating) {
        vlog("submitPrompt: keyboard send appears to have submitted", {
          iter,
          newCount,
          beforeKeyboardCount,
          generating,
        });
        keyboardSendOk = true;
        ready = true;
        break;
      }
    }

    if (Date.now() - lastShortLog > 2000) {
      vlog("submitPrompt: not yet ready", {
        iter,
        elapsedMs: Date.now() - startedAt,
        composerLen: length,
        sendVisible: !!sendInfo,
        sendDisabled: sendInfo ? sendInfo.disabled : null,
        sendAria: sendInfo ? sendInfo.aria : null,
        sendTestid: sendInfo ? sendInfo.testid : null,
        sendAriaDisabled: sendInfo ? sendInfo.ariaDisabled : null,
        generating,
        keyboardSendAttempted,
      });
      lastShortLog = Date.now();
    }

    if (Date.now() - lastVerboseLog > 15000) {
      await dumpComposerState(page, `submit-loop-iter-${iter}`);
      lastVerboseLog = Date.now();
    }

    // Backup #1 (DOM insert): if the composer is still empty after 30s,
    // ChatGPT didn't pick up the native paste -- try inserting via
    // execCommand.
    if (!domInsertAttempted && length === 0 && Date.now() - startedAt > 30000) {
      vlog("submitPrompt: native paste did not populate composer; trying DOM insert");
      domInsertAttempted = true;
      domInsertOk = await domInsertPrompt(page, fullPrompt);
      vlog("submitPrompt: DOM insert result", { domInsertOk });
      await page.waitForTimeout(1500);
      continue;
    }

    // Backup #2 (keyboard): if composer has text but the send button keeps
    // looking unclickable for >20s, try Cmd+Enter / Ctrl+Enter.
    if (
      !keyboardSendAttempted &&
      length > 0 &&
      Date.now() - startedAt > 20000 &&
      (!sendInfo || sendInfo.disabled)
    ) {
      vlog(
        "submitPrompt: send button still unclickable after composer has text; trying keyboard send fallback"
      );
      keyboardSendAttempted = await sendViaKeyboard(page);
      vlog("submitPrompt: keyboard send fallback initiated", { keyboardSendAttempted });
      await page.waitForTimeout(1500);
      continue;
    }

    await page.waitForTimeout(1000);
  }

  if (!ready) {
    const length = await composerTextLength(page);
    const sendInfo = await inspectSendButton(page);
    await dumpComposerState(page, "submit-timeout");
    throw new Error(
      `Prompt was not ready to send before timeout; composerLen=${length}, ` +
        `sendVisible=${!!sendInfo}, sendDisabled=${sendInfo ? sendInfo.disabled : "n/a"}, ` +
        `sendAria=${sendInfo ? JSON.stringify(sendInfo.aria) : "n/a"}, ` +
        `domInsertAttempted=${domInsertAttempted}, domInsertOk=${domInsertOk}, ` +
        `keyboardSendAttempted=${keyboardSendAttempted}, keyboardSendOk=${keyboardSendOk}`
    );
  }

  vlog("submitPrompt: send complete", {
    iter,
    elapsedMs: Date.now() - startedAt,
    keyboardSendOk,
    domInsertOk,
  });
}

async function waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs) {
  vlog("waitForNewAssistantToSettle: start", { beforeCount, responseTimeoutMs });
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
  vlog("newChatPage: navigating to chatgpt.com");
  await page.goto("https://chatgpt.com/", { waitUntil: "domcontentloaded", timeout: pageTimeoutMs });
  vlog("newChatPage: dom loaded, sleeping 3s for hydration");
  await page.waitForTimeout(3000);
  await findPromptBox(page, Math.min(pageTimeoutMs, 60000));
  vlog("newChatPage: prompt box visible", { url: page.url() });
  return page;
}

async function currentOrFreshChatPage(browser, pageTimeoutMs, composerWaitMs) {
  const candidates = await listChatGptPages(browser);
  vlog("currentOrFreshChatPage: candidate ChatGPT tabs", {
    count: candidates.length,
    urls: candidates.map((c) => c.url),
  });
  if (candidates.length === 0) {
    vlog("currentOrFreshChatPage: no candidates; opening fresh tab");
    return await newChatPage(browser, pageTimeoutMs);
  }
  for (const candidate of candidates.slice().reverse()) {
    const page = candidate.page;
    if (page.isClosed()) continue;
    try {
      await page.bringToFront();
      await page.waitForLoadState("domcontentloaded", { timeout: 5000 }).catch(() => {});
      if (await waitForComposerReadyForNextPrompt(page, composerWaitMs, "current-tab reuse")) {
        vlog("currentOrFreshChatPage: reusing tab", { url: page.url(), title: candidate.title });
        return page;
      }
    } catch (err) {
      vwarn("currentOrFreshChatPage: candidate threw", { err: err.message });
    }
  }
  vlog("currentOrFreshChatPage: no reusable tab; opening fresh tab");
  return await newChatPage(browser, pageTimeoutMs);
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
// Prompt wrapper
// ---------------------------------------------------------------------------
function buildWrappedPrompt(userPrompt, token) {
  return `
You are generating files for local automation.

Do NOT use ChatGPT's attachment UI, upload UI, microphone, voice mode, dictation, canvas, or special download button.

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
  });

  const repoRoot = resolveRepoRoot(args, outDir);
  vlog("repo root resolved", { repoRoot, marker: markerPath(repoRoot) });

  const userPrompt = readPromptFile(promptFile);
  const promptHash = sha256(userPrompt);
  vlog("prompt loaded", {
    file: promptFile,
    chars: userPrompt.length,
    hash: promptHash.slice(0, 16),
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
          : await currentOrFreshChatPage(browser, pageTimeoutMs, composerWaitMs);
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

      const result = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs);
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
      });
      return;
    }

    let beforeCount = Number(marker.beforeCount);
    if (!Number.isFinite(beforeCount) || beforeCount < 0) {
      const cur = await assistantCount(page);
      beforeCount = Math.max(0, cur - 1);
    }
    vlog("resume: waiting for assistant message to settle", { beforeCount });
    const result = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs);
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
}) {
  const answerPath = path.join(metaDir, "answer.md");
  fs.writeFileSync(answerPath, answer || "", "utf8");
  vlog("finalizeAnswer: wrote answer.md", { path: answerPath, len: (answer || "").length });

  const parsedFiles = parseFileBlocks(answer || "", token);
  vlog("finalizeAnswer: parsed file blocks", { count: parsedFiles.length });
  const { written, skipped } = writeGeneratedFiles(parsedFiles, outDir);
  vlog("finalizeAnswer: write summary", { written: written.length, skipped: skipped.length });

  if (written.length === 0 || skipped.length > 0) {
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

  if (settled) {
    clearMarker(repoRoot);
  } else {
    vlog(
      "settled=false; preserving in-flight marker so a follow-up invocation can resume waiting"
    );
  }

  const manifest = {
    ok: written.length > 0,
    settled: !!settled,
    mode:
      mode === "new"
        ? "new-chatgpt-tab-gptweb-file-parser"
        : "existing-or-fresh-chatgpt-tab-gptweb-file-parser",
    promptFile: path.resolve(promptFile),
    outDir,
    metaDir,
    answerPath,
    token,
    files: written,
    skipped,
    parsedBlockCount: parsedFiles.length,
    keptChatGptTabOpen: !closeTab,
    chatGptTabUrl: page.url(),
    note:
      written.length === 0
        ? "No files were written. Check .gpt-web-run/answer.md, debug.log, after-generation.png."
        : "Parsed GPTWEB_FILE blocks and wrote generated files directly under --out.",
  };
  fs.writeFileSync(path.join(metaDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  vlog("manifest written", manifest);

  if (!settled) {
    vlog("exit code 2 (settled=false)");
    process.exitCode = 2;
  } else {
    vlog("exit code 0 (settled=true)");
  }
}

module.exports = { run };
