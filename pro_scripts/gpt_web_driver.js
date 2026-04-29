const { chromium } = require("playwright");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const child_process = require("child_process");

const CDP_URL = process.env.GPT_WEB_CDP_URL || "http://127.0.0.1:9222";

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
      console.log(`Wrote file: ${dest}`);
    } catch (err) {
      skipped.push({ relPath: f.relPath, error: err.message });
    }
  }
  return { written, skipped };
}

// ---------------------------------------------------------------------------
// In-flight prompt marker.  The marker is the linchpin of robustness: as soon
// as we click "send", we record on disk that a prompt is in flight in a
// specific ChatGPT tab.  If this process is killed/timed-out before the
// response settles, the next invocation (typically current_tab.js) will see
// the marker, find the same tab, and RESUME WAITING instead of submitting the
// prompt a second time.  Sending a duplicate prompt while ChatGPT is still
// reasoning is the failure mode we are eliminating.
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
  // Atomic write: write to a sibling temp file and rename.  This prevents
  // half-written marker files if the process is killed mid-write -- a
  // truncated marker would otherwise look like a stale-but-valid entry.
  const tmp = `${p}.tmp.${process.pid}.${Date.now().toString(36)}`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), "utf8");
  fs.renameSync(tmp, p);
}

function clearMarker(repoRoot) {
  try {
    const p = markerPath(repoRoot);
    if (fs.existsSync(p)) fs.unlinkSync(p);
  } catch (_) {}
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

// After clicking send on a brand-new chat, ChatGPT navigates the tab from
// "/" to "/c/<convId>" within a few hundred ms.  Capturing the conversation
// URL lets the resume path identify the tab unambiguously even if the user
// has multiple ChatGPT tabs open.  Returns the conversation URL if it
// appears within `timeoutMs`, otherwise returns whatever the current URL
// is (caller should still write a marker; findTabByUrl tolerates this).
async function waitForConversationUrl(page, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const url = page.url();
    if (isConversationUrl(url)) return url;
    await page.waitForTimeout(200);
  }
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
async function findPromptBox(page, timeout = 10000) {
  const candidates = [
    '[data-testid="prompt-textarea"]',
    "#prompt-textarea",
    'div[contenteditable="true"][id="prompt-textarea"]',
    '.ProseMirror[contenteditable="true"]',
    'textarea[placeholder*="Message"]',
    "textarea",
    'div[contenteditable="true"]',
    '[contenteditable="true"]',
  ];
  for (const selector of candidates) {
    const loc = page.locator(selector).last();
    try {
      await loc.waitFor({ state: "visible", timeout });
      return loc;
    } catch (_) {}
  }
  throw new Error("Could not find ChatGPT prompt box.");
}

async function composerTextLength(page) {
  return await page
    .evaluate(() => {
      const selectors = [
        '[data-testid="prompt-textarea"]',
        "#prompt-textarea",
        '.ProseMirror[contenteditable="true"]',
        'textarea[placeholder*="Message"]',
        "textarea",
        'div[contenteditable="true"]',
        '[contenteditable="true"]',
      ];
      for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (!el) continue;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") continue;
        if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") return (el.value || "").length;
        return (el.innerText || el.textContent || "").length;
      }
      return 0;
    })
    .catch(() => 0);
}

async function clearPromptBoxInDom(page) {
  await page
    .evaluate(() => {
      const selectors = [
        '[data-testid="prompt-textarea"]',
        "#prompt-textarea",
        '.ProseMirror[contenteditable="true"]',
        'textarea[placeholder*="Message"]',
        "textarea",
        'div[contenteditable="true"]',
        '[contenteditable="true"]',
      ];
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
    })
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

// Detect whether ChatGPT is currently generating a response.  Conservative
// by design -- a false positive here causes the driver to wait forever for
// the composer to become "ready", which manifests as "nothing happens".
//
// We rely on two narrow, unambiguous signals:
//   1. A visible Stop button.  This is the canonical signal during ordinary
//      streaming and is set by ChatGPT itself.
//   2. A data-testid attribute that explicitly contains "thinking" or
//      "reasoning".  ChatGPT's reasoning models render a "Thinking" pill
//      with a stable testid before the first token appears.
//
// We deliberately do NOT scan free text on the page (aria-live regions,
// arbitrary innerText) because the homepage and many idle states contain
// words like "working", "reading", "planning" in unrelated contexts.
async function isGenerating(page) {
  const stopSelectors = [
    'button[data-testid="stop-button"]',
    'button[aria-label="Stop streaming"]',
    'button[aria-label="Stop generating"]',
    'button[aria-label="Stop generating response"]',
  ];
  for (const selector of stopSelectors) {
    try {
      const loc = page.locator(selector).last();
      if ((await loc.count()) > 0 && (await loc.isVisible().catch(() => false))) return true;
    } catch (_) {}
  }

  // Reasoning pill: ChatGPT marks its "Thinking..." indicator with a
  // data-testid that contains "thinking" (and on some builds "reasoning").
  // We require the testid match -- not a free-text match -- to keep this
  // signal narrow.
  try {
    const reasoningTestid = page
      .locator('[data-testid*="thinking" i], [data-testid*="reasoning" i]')
      .last();
    if ((await reasoningTestid.count()) > 0 && (await reasoningTestid.isVisible().catch(() => false))) {
      return true;
    }
  } catch (_) {}

  return false;
}

async function findSendButton(page) {
  const selectors = [
    'button[data-testid="send-button"]',
    'button[aria-label="Send prompt"]',
    'button[aria-label="Send message"]',
    'button[aria-label*="Send"]',
  ];
  for (const selector of selectors) {
    const loc = page.locator(selector).last();
    const count = await loc.count().catch(() => 0);
    if (count === 0) continue;
    if (await loc.isVisible().catch(() => false)) return loc;
  }
  return null;
}

async function waitForComposerReadyForNextPrompt(page, timeoutMs = 120000, label = "next prompt") {
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
      console.log(`Composer is visible for ${label}.`);
      return true;
    }
    if (Date.now() - lastLog > 10000) {
      console.log(
        `Waiting for composer ready for ${label} (composerVisible=${composerVisible}, ` +
          `generating=${generating}, iter=${iter}, ` +
          `remainingMs=${Math.max(0, deadline - Date.now())}).`
      );
      lastLog = Date.now();
    }
    await page.waitForTimeout(1000);
  }
  console.log(`Composer did not become visible for ${label} within ${timeoutMs}ms.`);
  return false;
}

// ---------------------------------------------------------------------------
// Clipboard / paste helpers
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
  // Non-macOS fallback. On Linux/Windows this should still be a plain paste in
  // most browser editors.
  throw new Error("native plain paste is only implemented through osascript on macOS");
}

async function domInsertPrompt(page, fullPrompt) {
  return await page
    .evaluate((text) => {
      const selectors = [
        '[data-testid="prompt-textarea"]',
        "#prompt-textarea",
        '.ProseMirror[contenteditable="true"]',
        'textarea[placeholder*="Message"]',
        "textarea",
        'div[contenteditable="true"]',
        '[contenteditable="true"]',
      ];
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
    }, fullPrompt)
    .catch(() => false);
}

// ---------------------------------------------------------------------------
// Submit / wait core
// ---------------------------------------------------------------------------
async function submitPrompt(page, fullPrompt, promptTimeoutMs) {
  await page.bringToFront().catch(() => {});
  const promptBox = await findPromptBox(page, Math.min(promptTimeoutMs, 60000));
  await promptBox.scrollIntoViewIfNeeded().catch(() => {});
  await promptBox.click({ timeout: 10000 }).catch(() => {});
  await clearPromptBoxInDom(page);
  await page.waitForTimeout(300);

  setClipboardText(fullPrompt);
  console.log(
    `Pasting prompt as plain text using native Cmd+Shift+V (${fullPrompt.length} characters).`
  );
  try {
    nativePlainTextPaste();
  } catch (err) {
    console.log(`Native paste failed: ${err.message}`);
  }

  const deadline = Date.now() + promptTimeoutMs;
  let lastLog = 0;
  let ready = false;
  while (Date.now() < deadline) {
    const length = await composerTextLength(page);
    const sendButton = await findSendButton(page);
    const disabled = sendButton ? await sendButton.isDisabled().catch(() => true) : true;
    if (length > 0 && sendButton && !disabled) {
      console.log(`Prompt entry ready: composer length=${length}.`);
      await sendButton.click({ timeout: 10000 });
      ready = true;
      break;
    }
    if (Date.now() - lastLog > 10000) {
      console.log(`Waiting for ChatGPT composer/send button after paste; composer length=${length}.`);
      lastLog = Date.now();
    }
    // Give native paste a real chance before any DOM fallback. With large prompts
    // ChatGPT may need several seconds to populate the editor.
    if (Date.now() > deadline - promptTimeoutMs + 45000 && length === 0) {
      console.log("Native paste has not populated the composer; trying scoped DOM insert fallback.");
      const ok = await domInsertPrompt(page, fullPrompt);
      if (ok) await page.waitForTimeout(1500);
    }
    await page.waitForTimeout(1000);
  }

  if (!ready) {
    const length = await composerTextLength(page);
    throw new Error(`Prompt was not ready to send before timeout; composer length=${length}`);
  }
}

async function waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs) {
  let previous = "";
  let stableCount = 0;
  let lastNonEmpty = "";
  let sawNewMessage = false;
  let lastLog = 0;
  const deadline = Date.now() + responseTimeoutMs;

  while (Date.now() < deadline) {
    await page.waitForTimeout(1000);
    const loc = page.locator('[data-message-author-role="assistant"]');
    const count = await loc.count().catch(() => 0);
    let current = "";
    if (count > beforeCount) {
      sawNewMessage = true;
      current = await loc.nth(count - 1).innerText().catch(() => "");
    } else {
      // Even before a new assistant block exists (during long reasoning), we
      // may want to peek at the trailing assistant text for partial output.
      current = await lastAssistantText(page);
    }
    current = current.trim();
    if (current) lastNonEmpty = current;

    const generating = await isGenerating(page);

    if (Date.now() - lastLog > 30000) {
      const elapsedMs = responseTimeoutMs - (deadline - Date.now());
      console.log(
        `Waiting for response to settle (elapsed=${Math.round(elapsedMs / 1000)}s, ` +
          `assistantCount=${count}, sawNew=${sawNewMessage}, generating=${generating}, ` +
          `currentLen=${current.length}, stable=${stableCount}).`
      );
      lastLog = Date.now();
    }

    if (current && current === previous && !generating && sawNewMessage) {
      stableCount++;
      if (stableCount >= 5) {
        await waitForComposerReadyForNextPrompt(page, 120000, "follow-up prompt after response");
        return { text: current, settled: true, sawNewMessage };
      }
    } else {
      stableCount = 0;
      previous = current;
    }
  }
  console.log(
    `Response did not settle within ${responseTimeoutMs}ms (sawNew=${sawNewMessage}, lastLen=${lastNonEmpty.length}).`
  );
  await waitForComposerReadyForNextPrompt(page, 30000, "follow-up prompt after response timeout");
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
}

async function connectBrowser(connectTimeoutMs) {
  return await chromium.connectOverCDP(CDP_URL, { timeout: connectTimeoutMs });
}

async function newChatPage(browser, pageTimeoutMs) {
  const context = browser.contexts()[0];
  if (!context) throw new Error(`No existing Chrome context found at ${CDP_URL}`);
  context.setDefaultTimeout(Math.min(pageTimeoutMs, 30000));
  const page = await context.newPage();
  await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
  await page.goto("https://chatgpt.com/", { waitUntil: "domcontentloaded", timeout: pageTimeoutMs });
  await page.waitForTimeout(3000);
  await findPromptBox(page, Math.min(pageTimeoutMs, 60000));
  return page;
}

// Find the most recent ChatGPT tab to reuse.  Unlike the previous version,
// we DO NOT clone the conversation URL into a brand-new tab when the
// composer is missing -- cloning combined with the Python orchestrator's
// retry logic is what caused duplicate prompts to be submitted while
// ChatGPT was still reasoning.  Instead we simply wait, with a generous
// timeout, for the original tab to become ready, or open a fresh one.
async function currentOrFreshChatPage(browser, pageTimeoutMs, composerWaitMs) {
  const candidates = await listChatGptPages(browser);
  if (candidates.length === 0) {
    console.log("No open ChatGPT tab found; opening a fresh ChatGPT tab.");
    return await newChatPage(browser, pageTimeoutMs);
  }

  for (const candidate of candidates.slice().reverse()) {
    const page = candidate.page;
    if (page.isClosed()) continue;
    try {
      await page.bringToFront();
      await page.waitForLoadState("domcontentloaded", { timeout: 5000 }).catch(() => {});
      if (await waitForComposerReadyForNextPrompt(page, composerWaitMs, "current tab reuse")) {
        console.log(`Using existing ChatGPT tab: ${candidate.title || "(untitled)"}`);
        console.log(`URL: ${page.url()}`);
        return page;
      }
    } catch (_) {}
  }

  console.log("No reusable ChatGPT tab had a ready composer; opening a fresh ChatGPT tab.");
  return await newChatPage(browser, pageTimeoutMs);
}

// Locate a tab by exact URL match (used during marker resume).  Returns null
// if not found.  We try (in order):
//   1. Exact URL match (best when user has multiple ChatGPT tabs)
//   2. Conversation-id match (handles the case where the tab's URL has been
//      updated since the marker was written, e.g. user navigated within the
//      same conversation, or our captured URL was already a conv URL)
//   3. If the marker URL is not a conversation URL AND there is exactly
//      ONE open ChatGPT tab, use that tab.  This covers the brief window
//      after a brand-new send where the tab URL is still "/" instead of
//      "/c/<id>" -- there's no ambiguity in that case.
async function findTabByUrl(browser, targetUrl) {
  if (!targetUrl) return null;
  const targetConv = conversationIdFromUrl(targetUrl);
  const pages = await listChatGptPages(browser);
  // Exact URL match first.
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
  // Last resort: if the marker's URL is a non-conversation URL (typically
  // "/" captured before ChatGPT redirected) and there is exactly one open
  // ChatGPT tab, that tab is unambiguously the one we sent the prompt to.
  if (!isConversationUrl(targetUrl)) {
    const open = pages.filter((c) => !c.page.isClosed());
    if (open.length === 1) {
      console.log(
        `Marker URL (${targetUrl}) is not a conversation URL, but there is exactly one ` +
          `open ChatGPT tab (${open[0].url}); using it for resume.`
      );
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

// ---------------------------------------------------------------------------
// Repo root resolution.  The marker lives at <repoRoot>/.vc4_auto/, but this
// driver is invoked with --out pointing at .vc4_auto/staging/<test>/<attempt>.
// We prefer an explicit --repo-root, otherwise we walk up from --out looking
// for .vc4_auto, otherwise we fall back to cwd.
// ---------------------------------------------------------------------------
function resolveRepoRoot(args, outDir) {
  if (args["repo-root"]) {
    const explicit = path.resolve(args["repo-root"]);
    return explicit;
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
  // 30 minutes of model wallclock by default.  The Python orchestrator passes
  // its own value here; this is just the standalone fallback.
  const responseTimeoutMs = intArg(args, "response-timeout-ms", 30 * 60 * 1000);
  const composerWaitMs = intArg(args, "composer-wait-ms", 5 * 60 * 1000);
  const allowDuplicateSubmit = boolArg(args, "allow-duplicate-submit");

  console.log(
    `[gpt_web_driver] starting mode=${mode} promptFile=${promptFile} outDir=${outDir}`
  );
  console.log(
    `[gpt_web_driver] timeouts: connect=${connectTimeoutMs}ms page=${pageTimeoutMs}ms ` +
      `prompt=${promptTimeoutMs}ms response=${responseTimeoutMs}ms composerWait=${composerWaitMs}ms`
  );

  mkdirp(outDir);
  mkdirp(metaDir);

  const repoRoot = resolveRepoRoot(args, outDir);
  console.log(`[gpt_web_driver] resolved repo root for in-flight marker: ${repoRoot}`);

  const userPrompt = readPromptFile(promptFile);
  const promptHash = sha256(userPrompt);
  console.log(
    `[gpt_web_driver] prompt loaded: ${userPrompt.length} chars, hash=${promptHash.slice(0, 12)}`
  );

  console.log(`[gpt_web_driver] connecting to Chrome via CDP at ${CDP_URL}...`);
  const browser = await connectBrowser(connectTimeoutMs);
  console.log(`[gpt_web_driver] connected to Chrome.`);
  let page = null;
  let resumed = false;
  let token = null;
  let fullPrompt = null;

  try {
    await denyMicIfPossible(browser);

    // ---- Resume path: an in-flight marker for the same prompt hash exists ----
    const marker = readMarker(repoRoot);
    if (marker && marker.promptHash === promptHash && marker.tabUrl && marker.token) {
      const ageMs = Date.now() - Number(marker.submittedAt || 0);
      console.log(
        `Found in-flight marker: tabUrl=${marker.tabUrl}, ageMs=${ageMs}, token=${marker.token}`
      );
      const existing = await findTabByUrl(browser, marker.tabUrl);
      if (existing) {
        console.log(
          "Reusing existing ChatGPT tab to RESUME WAITING on the previously-submitted prompt. " +
            "No new prompt will be sent."
        );
        page = existing;
        token = marker.token;
        fullPrompt = null;
        resumed = true;
        await page.bringToFront().catch(() => {});
      } else {
        console.log(
          "In-flight marker present but the matching ChatGPT tab is gone; discarding marker."
        );
        clearMarker(repoRoot);
      }
    } else if (marker && marker.promptHash !== promptHash) {
      // Different prompt -- the previous run is unrelated to this one.  Leave
      // the marker alone so a future invocation with the matching hash can
      // still resume; just don't try to resume here.
      console.log(
        `In-flight marker exists for a different prompt hash (have ${marker.promptHash}, want ${promptHash}). ` +
          "Proceeding with normal submit; the marker will be replaced."
      );
    }

    // ---- Normal path: no resume -> open tab and submit ----
    if (!resumed) {
      console.log(`[gpt_web_driver] opening ${mode === "new" ? "new" : "current/fresh"} ChatGPT tab...`);
      page =
        mode === "new"
          ? await newChatPage(browser, pageTimeoutMs)
          : await currentOrFreshChatPage(browser, pageTimeoutMs, composerWaitMs);
      console.log(`[gpt_web_driver] tab ready: ${page.url()}`);

      const context = page.context();
      context.setDefaultTimeout(Math.min(pageTimeoutMs, 30000));
      await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
      await page.bringToFront();

      // Refuse to submit if a different in-flight prompt is recorded and the
      // user hasn't explicitly waived the safety.  This protects against the
      // duplicate-submit failure mode the orchestrator used to trigger.
      const existingMarker = readMarker(repoRoot);
      if (existingMarker && existingMarker.promptHash !== promptHash && !allowDuplicateSubmit) {
        const existingPage = await findTabByUrl(browser, existingMarker.tabUrl);
        if (existingPage && (await isGenerating(existingPage).catch(() => false))) {
          throw new Error(
            "Refusing to submit a new prompt while a different in-flight prompt is still " +
              "generating in another ChatGPT tab. Wait for it to finish, clear " +
              `${markerPath(repoRoot)}, or pass --allow-duplicate-submit.`
          );
        }
      }

      // For a freshly-opened tab, newChatPage already verified the composer
      // is visible.  For an existing tab, currentOrFreshChatPage already
      // waited for composer-ready.  Do a brief 10s composer-visibility
      // sanity check here -- but DO NOT loop on isGenerating, because
      // isGenerating is intentionally conservative and we only need the
      // composer to be present and clickable.
      try {
        await findPromptBox(page, 10000);
      } catch (err) {
        throw new Error(`Prompt box not visible before submit: ${err.message}`);
      }

      token = randomToken();
      fullPrompt = buildWrappedPrompt(userPrompt, token);

      const beforeCount = await assistantCount(page);
      await submitPrompt(page, fullPrompt, promptTimeoutMs);
      // After the send click, ChatGPT navigates the tab from "/" to
      // "/c/<convId>" once the conversation is created server-side.  This
      // happens within a few hundred ms in normal cases but can take a few
      // seconds.  We want the marker to capture the conversation URL so a
      // resume invocation can find the right tab even if the user has
      // switched tabs around.  Wait up to 15s for the URL to become a
      // conversation URL; if it doesn't, fall back to the current URL --
      // findTabByUrl tolerates non-conversation URLs in single-tab cases.
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
      console.log(
        `Prompt submitted in ${mode === "new" ? "new" : "current"} ChatGPT tab. ` +
          `In-flight marker written (tabUrl=${tabUrl}). Waiting for response to settle...`
      );

      // Settle wait.
      const result = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs);
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

    // ---- Resume branch: keep waiting on the existing tab ----
    // We don't know `beforeCount` from the original submission with full
    // certainty (the marker stores it, but the count may have advanced if
    // ChatGPT already produced text).  Use the stored value when available,
    // otherwise fall back to "current count - 1" so we still treat the most
    // recent assistant block as the response under construction.
    let beforeCount = Number(marker.beforeCount);
    if (!Number.isFinite(beforeCount) || beforeCount < 0) {
      const cur = await assistantCount(page);
      beforeCount = Math.max(0, cur - 1);
    }
    console.log(`Resume: waiting for assistant message #${beforeCount + 1} to settle.`);
    const result = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs);
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
  const parsedFiles = parseFileBlocks(answer || "", token);
  const { written, skipped } = writeGeneratedFiles(parsedFiles, outDir);

  if (written.length === 0 || skipped.length > 0) {
    await dumpDebugState(page, metaDir);
  } else {
    if (
      !(await waitForComposerReadyForNextPrompt(
        page,
        60000,
        "next automation step after file parse"
      ))
    ) {
      const restoreUrl = page.url();
      if (isChatGptUrl(restoreUrl)) {
        console.log(
          `Composer still hidden; reloading current chat before returning control: ${restoreUrl}`
        );
        await page
          .goto(restoreUrl, { waitUntil: "domcontentloaded", timeout: pageTimeoutMs })
          .catch(() => {});
        await page.waitForTimeout(4000);
        await waitForComposerReadyForNextPrompt(page, 60000, "next automation step after reload");
      }
    }
  }

  // Marker lifecycle:
  //   - If the response settled (settled=true), the prompt is "done" --
  //     either successfully (files written) or unrecoverably (model returned
  //     prose instead of GPTWEB_FILE blocks).  Either way, a future retry
  //     should NOT resume waiting on this prompt.  Clear the marker.
  //   - If the wait did NOT settle (settled=false), ChatGPT is probably still
  //     reasoning when our internal timeout fired.  Leave the marker on disk
  //     so the next invocation can resume waiting on the same tab instead of
  //     re-submitting the prompt.
  if (settled) {
    clearMarker(repoRoot);
  } else {
    console.log(
      "Wait timed out without a settled response; leaving the in-flight marker so a follow-up " +
        "invocation can resume waiting on the same ChatGPT tab."
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
        ? "No files were written. Check .gpt-web-run/answer.md and .gpt-web-run/after-generation.png."
        : "Parsed GPTWEB_FILE blocks and wrote generated files directly under --out.",
  };
  fs.writeFileSync(path.join(metaDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  console.log(JSON.stringify(manifest, null, 2));
  // Exit code semantics, deliberately distinguishing between "infra failure
  // (Python should retry the same prompt, possibly resuming via the marker)"
  // and "model produced an answer that the orchestrator can act on
  // (Python should validate via copy_staged_files_into_repo and, if
  // empty/incomplete, run its chat-output-validation -> fix prompt flow)":
  //
  //   - exit 0: response settled.  Whether or not files were written, the
  //     orchestrator owns the next decision based on the staging dir.
  //   - exit 2: response did NOT settle.  This is an infra-level failure
  //     and the orchestrator should treat it as such (the in-flight marker
  //     has been preserved so a retry will resume waiting on the same tab).
  if (!settled) {
    process.exitCode = 2;
  }
}

module.exports = { run };
