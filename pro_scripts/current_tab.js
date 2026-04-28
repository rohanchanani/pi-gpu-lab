#!/usr/bin/env node

const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");
const child_process = require("child_process");

const CDP_URL = process.env.GPT_WEB_CDP_URL || "http://127.0.0.1:9222";

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const key = argv[i];
    if (!key.startsWith("--")) continue;
    const name = key.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) args[name] = "true";
    else {
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
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.floor(n);
}

function requireArg(args, name) {
  if (!args[name]) throw new Error(`Missing required argument: --${name}`);
  return args[name];
}

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
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")) || (t.startsWith("`") && t.endsWith("`"))) return t.slice(1, -1);
  return t;
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
  if (normalized === ".gpt-web-run" || normalized.startsWith(".gpt-web-run/")) throw new Error(`Refusing to write into reserved metadata directory: ${normalized}`);
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
      if (!(dest === root || dest.startsWith(root + path.sep))) throw new Error(`Refusing to write outside --out: ${relPath}`);
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

function isChatGptUrl(url) {
  return /^https?:\/\/(chatgpt\.com|chat\.openai\.com)(\/|$)/i.test(url || "");
}

function chatCloneUrl(url) {
  const u = String(url || "");
  const m = u.match(/^https?:\/\/(?:chatgpt\.com|chat\.openai\.com)\/c\/([^/?#]+)/i);
  if (m) return `https://chatgpt.com/c/${m[1]}`;
  return "https://chatgpt.com/";
}

const PROMPT_SELECTORS = [
  '[data-testid="prompt-textarea"]',
  '#prompt-textarea',
  'div[contenteditable="true"][id="prompt-textarea"]',
  '.ProseMirror[contenteditable="true"]',
  'textarea[placeholder*="Message"]',
  'textarea',
  'div[contenteditable="true"]',
  '[contenteditable="true"]',
];

const SEND_SELECTORS = [
  'button[data-testid="send-button"]',
  '#composer-submit-button',
  'button[aria-label="Send prompt"]',
  'button[aria-label="Send message"]',
  'button[aria-label*="Send"]',
];

async function findPromptBox(page, timeout = 10000) {
  const perSelector = Math.max(1000, Math.floor(timeout / PROMPT_SELECTORS.length));
  for (const selector of PROMPT_SELECTORS) {
    const loc = page.locator(selector).last();
    try {
      await loc.waitFor({ state: "visible", timeout: perSelector });
      return loc;
    } catch (_) {}
  }
  throw new Error("Could not find ChatGPT prompt box on the selected tab.");
}

async function waitForPromptBox(page, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastErr = null;
  while (Date.now() < deadline) {
    try {
      return await findPromptBox(page, Math.min(10000, Math.max(1000, deadline - Date.now())));
    } catch (err) {
      lastErr = err;
      await page.waitForTimeout(1000).catch(() => {});
    }
  }
  throw lastErr || new Error("Could not find ChatGPT prompt box before timeout.");
}

async function listChatGptPages(browser) {
  const pages = [];
  for (const context of browser.contexts()) {
    for (const page of context.pages()) {
      const url = page.url();
      if (!isChatGptUrl(url)) continue;
      let title = "";
      try { title = await page.title(); } catch (_) {}
      pages.push({ context, page, url, title });
    }
  }
  return pages;
}

async function cloneChatTab(candidate, pageTimeoutMs, promptTimeoutMs) {
  const oldUrl = candidate.url || candidate.page.url();
  const cloneUrl = chatCloneUrl(oldUrl);
  console.log(`Prompt box missing/hidden on current ChatGPT tab; opening clone: ${cloneUrl}`);
  const page = await candidate.context.newPage();
  await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
  await page.goto(cloneUrl, { waitUntil: "domcontentloaded", timeout: pageTimeoutMs });
  await page.waitForLoadState("networkidle", { timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(1500);
  await page.bringToFront().catch(() => {});
  await waitForPromptBox(page, promptTimeoutMs);
  return page;
}

async function findExistingOrClonedChatGptPage(browser, pageTimeoutMs, promptTimeoutMs) {
  const candidates = await listChatGptPages(browser);
  if (candidates.length === 0) throw new Error(`No open ChatGPT tab found. Open https://chatgpt.com/ in Chrome running at ${CDP_URL}, then rerun.`);

  const reversed = candidates.slice().reverse();
  let firstUsable = null;
  for (const candidate of reversed) {
    const page = candidate.page;
    if (page.isClosed()) continue;
    if (!firstUsable) firstUsable = candidate;
    try {
      await page.bringToFront();
      await page.waitForLoadState("domcontentloaded", { timeout: 5000 }).catch(() => {});
      await findPromptBox(page, 5000);
      console.log(`Using existing ChatGPT tab: ${candidate.title || "(untitled)"}`);
      console.log(`URL: ${candidate.url}`);
      return page;
    } catch (_) {}
  }

  if (firstUsable) {
    return await cloneChatTab(firstUsable, pageTimeoutMs, promptTimeoutMs);
  }

  throw new Error("Found ChatGPT tab(s), but none could be used or cloned.");
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

async function waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs) {
  let previous = "";
  let stableCount = 0;
  let lastNonEmpty = "";
  const deadline = Date.now() + responseTimeoutMs;
  while (Date.now() < deadline) {
    await page.waitForTimeout(1000);
    const loc = page.locator('[data-message-author-role="assistant"]');
    const count = await loc.count().catch(() => 0);
    let current = "";
    if (count > beforeCount) current = await loc.nth(count - 1).innerText().catch(() => "");
    else if (Date.now() > deadline - responseTimeoutMs + 20000) current = await lastAssistantText(page);
    current = current.trim();
    if (current) lastNonEmpty = current;
    if (current && current === previous) {
      stableCount++;
      if (stableCount >= 5) return current;
    } else {
      stableCount = 0;
      previous = current;
    }
  }
  return lastNonEmpty;
}

function setClipboardText(text) {
  if (process.platform === "darwin") {
    const res = child_process.spawnSync("/usr/bin/pbcopy", [], { input: text, encoding: "utf8", maxBuffer: Math.max(1024 * 1024, Buffer.byteLength(text, "utf8") + 1024) });
    if (res.status !== 0) throw new Error(`pbcopy failed: ${res.stderr || res.stdout || res.status}`);
    return;
  }
  throw new Error("clipboard paste helper currently requires macOS pbcopy");
}

function runOsascript(script) {
  const res = child_process.spawnSync("/usr/bin/osascript", ["-e", script], { encoding: "utf8", timeout: 10000 });
  if (res.status !== 0) throw new Error(`osascript failed: ${res.stderr || res.stdout || res.status}`);
}

async function composerTextLength(page) {
  return await page.evaluate(() => {
    const selectors = ['[data-testid="prompt-textarea"]', '#prompt-textarea', '.ProseMirror[contenteditable="true"]', 'textarea', '[contenteditable="true"]'];
    for (const sel of selectors) {
      const els = Array.from(document.querySelectorAll(sel));
      const el = els[els.length - 1];
      if (!el) continue;
      const text = el.tagName === 'TEXTAREA' ? (el.value || '') : (el.innerText || el.textContent || '');
      return text.length;
    }
    return 0;
  }).catch(() => 0);
}

async function clearComposerDom(page) {
  await page.evaluate(() => {
    const selectors = ['[data-testid="prompt-textarea"]', '#prompt-textarea', '.ProseMirror[contenteditable="true"]', 'textarea', '[contenteditable="true"]'];
    const el = selectors.flatMap((sel) => Array.from(document.querySelectorAll(sel))).filter(Boolean).pop();
    if (!el) return;
    el.focus();
    if (el.tagName === 'TEXTAREA') el.value = '';
    else { el.textContent = ''; el.innerHTML = ''; }
    el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward' }));
  }).catch(() => {});
}

async function focusPromptBox(page, promptBox) {
  await page.bringToFront().catch(() => {});
  await promptBox.scrollIntoViewIfNeeded().catch(() => {});
  await promptBox.click({ timeout: 10000 }).catch(async () => {
    const box = await promptBox.boundingBox().catch(() => null);
    if (box) await page.mouse.click(box.x + Math.min(20, box.width / 2), box.y + Math.min(20, box.height / 2));
    else throw new Error("Could not click prompt box");
  });
}

async function nativePlainPaste(page, promptBox, fullPrompt) {
  setClipboardText(fullPrompt);
  await focusPromptBox(page, promptBox);
  await clearComposerDom(page);
  await focusPromptBox(page, promptBox);
  if (process.platform === "darwin") {
    console.log(`Pasting prompt as plain text using native Cmd+Shift+V (${fullPrompt.length} characters).`);
    await page.bringToFront().catch(() => {});
    // Do not use Cmd+A/Backspace. If focus is wrong, that can select/affect the whole page.
    runOsascript('tell application "System Events" to keystroke "v" using {command down, shift down}');
  } else {
    console.log(`Pasting prompt as plain text with Ctrl+Shift+V (${fullPrompt.length} characters).`);
    await page.keyboard.press("Control+Shift+V");
  }
}

async function domInjectPrompt(page, fullPrompt) {
  console.log("Native paste did not populate the composer; using DOM insertion fallback.");
  await page.evaluate((prompt) => {
    const selectors = ['[data-testid="prompt-textarea"]', '#prompt-textarea', '.ProseMirror[contenteditable="true"]', 'textarea', '[contenteditable="true"]'];
    const el = selectors.flatMap((sel) => Array.from(document.querySelectorAll(sel))).filter(Boolean).pop();
    if (!el) throw new Error("No composer element found for DOM insertion");
    el.focus();
    if (el.tagName === 'TEXTAREA') el.value = prompt;
    else el.textContent = prompt;
    el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: prompt.slice(0, 1) }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, fullPrompt);
}

async function findSendButton(page) {
  for (const selector of SEND_SELECTORS) {
    const loc = page.locator(selector).last();
    const count = await loc.count().catch(() => 0);
    if (count === 0) continue;
    try {
      await loc.waitFor({ state: "visible", timeout: 1000 });
      return loc;
    } catch (_) {}
  }
  return null;
}

async function sendButtonEnabled(page) {
  const btn = await findSendButton(page);
  if (!btn) return false;
  const disabled = await btn.isDisabled().catch(() => true);
  const ariaDisabled = await btn.getAttribute("aria-disabled").catch(() => null);
  return !disabled && ariaDisabled !== "true";
}

async function waitForComposerReadyToSend(page, promptTimeoutMs) {
  const deadline = Date.now() + promptTimeoutMs;
  let lastLog = 0;
  while (Date.now() < deadline) {
    const len = await composerTextLength(page);
    const enabled = await sendButtonEnabled(page);
    if (enabled && len > 0) {
      console.log(`Prompt entry ready: composer length=${len}.`);
      return true;
    }
    if (Date.now() - lastLog > 5000) {
      console.log(`Waiting for ChatGPT composer/send button; composer length=${len}.`);
      lastLog = Date.now();
    }
    await page.waitForTimeout(500);
  }
  return false;
}

async function clickSend(page) {
  const btn = await findSendButton(page);
  if (btn && (await sendButtonEnabled(page))) {
    await btn.click({ timeout: 10000 });
    return;
  }
  await page.keyboard.press("Enter");
}

async function submitPrompt(page, fullPrompt, promptTimeoutMs) {
  let promptBox = await waitForPromptBox(page, promptTimeoutMs);
  await nativePlainPaste(page, promptBox, fullPrompt);
  if (!(await waitForComposerReadyToSend(page, Math.min(promptTimeoutMs, 90000)))) {
    promptBox = await waitForPromptBox(page, Math.min(promptTimeoutMs, 30000));
    await domInjectPrompt(page, fullPrompt);
    if (!(await waitForComposerReadyToSend(page, Math.min(promptTimeoutMs, 90000)))) {
      throw new Error("Prompt text was not accepted into the ChatGPT composer before timeout.");
    }
  }
  await clickSend(page);
}

async function denyMicIfPossible(browser) {
  try {
    const session = await browser.newBrowserCDPSession();
    for (const permissionName of ["audioCapture", "microphone"]) {
      try { await session.send("Browser.setPermission", { permission: { name: permissionName }, setting: "denied", origin: "https://chatgpt.com" }); } catch (_) {}
    }
  } catch (_) {}
}

async function dumpDebugState(page, metaDir) {
  await page.screenshot({ path: path.join(metaDir, "after-generation.png"), fullPage: true }).catch(() => {});
  const candidates = await page.evaluate(() => {
    const els = [...document.querySelectorAll("a, button, textarea, [contenteditable='true']")];
    return els.map((el, i) => {
      const rect = el.getBoundingClientRect();
      return {
        i,
        tag: el.tagName.toLowerCase(),
        text: (el.innerText || el.textContent || el.value || "").trim().slice(0, 250),
        aria: el.getAttribute("aria-label") || "",
        title: el.getAttribute("title") || "",
        href: el.getAttribute("href") || "",
        testid: el.getAttribute("data-testid") || "",
        id: el.getAttribute("id") || "",
        className: String(el.getAttribute("class") || "").slice(0, 250),
        visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
        rect: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) },
      };
    });
  }).catch(() => []);
  fs.writeFileSync(path.join(metaDir, "click-candidates.json"), JSON.stringify(candidates, null, 2), "utf8");
}

async function disconnectBrowser(browser) {
  if (browser && typeof browser.disconnect === "function") await browser.disconnect().catch(() => {});
  else if (browser) await browser.close().catch(() => {});
}

async function main() {
  const args = parseArgs(process.argv);
  const promptFile = requireArg(args, "prompt-file");
  const outDir = path.resolve(requireArg(args, "out"));
  const metaDir = path.join(outDir, ".gpt-web-run");

  const connectTimeoutMs = intArg(args, "connect-timeout-ms", 120000);
  const pageTimeoutMs = intArg(args, "page-timeout-ms", 120000);
  const promptTimeoutMs = intArg(args, "prompt-timeout-ms", 120000);
  const responseTimeoutMs = intArg(args, "response-timeout-ms", 1200000);

  mkdirp(outDir);
  mkdirp(metaDir);

  const userPrompt = readPromptFile(promptFile);
  const token = randomToken();
  const fullPrompt = `
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

  const browser = await chromium.connectOverCDP(CDP_URL, { timeout: connectTimeoutMs });

  try {
    await denyMicIfPossible(browser);
    for (const context of browser.contexts()) context.setDefaultTimeout(pageTimeoutMs);
    const page = await findExistingOrClonedChatGptPage(browser, pageTimeoutMs, promptTimeoutMs);
    await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
    await page.bringToFront();

    const beforeCount = await assistantCount(page);
    await submitPrompt(page, fullPrompt, promptTimeoutMs);

    console.log("Prompt submitted into ChatGPT tab. Waiting for response to settle...");
    const answer = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs);
    const answerPath = path.join(metaDir, "answer.md");
    fs.writeFileSync(answerPath, answer || "", "utf8");

    const parsedFiles = parseFileBlocks(answer || "", token);
    const { written, skipped } = writeGeneratedFiles(parsedFiles, outDir);
    await dumpDebugState(page, metaDir);

    const manifest = {
      ok: written.length > 0,
      mode: "current-or-cloned-chatgpt-tab-gptweb-file-parser",
      promptFile: path.resolve(promptFile),
      outDir,
      metaDir,
      answerPath,
      token,
      files: written,
      skipped,
      parsedBlockCount: parsedFiles.length,
      chatGptTabUrl: page.url(),
      note: written.length === 0 ? "No files were written. Check .gpt-web-run/answer.md and .gpt-web-run/after-generation.png." : "Parsed GPTWEB_FILE blocks and wrote generated files directly under --out.",
    };
    fs.writeFileSync(path.join(metaDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
    console.log(JSON.stringify(manifest, null, 2));
    if (written.length === 0) process.exitCode = 2;
  } finally {
    await disconnectBrowser(browser);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
