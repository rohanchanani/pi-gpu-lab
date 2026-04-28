#!/usr/bin/env node

const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");
const childProcess = require("child_process");

const CDP_URL = process.env.GPT_WEB_CDP_URL || "http://127.0.0.1:9222";

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

function intArg(args, name, defaultValue) {
  const raw = args[name];
  if (raw === undefined || raw === null || raw === "") return defaultValue;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`Invalid positive integer for --${name}: ${raw}`);
  }
  return Math.floor(n);
}

function requireArg(args, name) {
  if (!args[name]) {
    throw new Error(`Missing required argument: --${name}`);
  }
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
  if (
    (t.startsWith('"') && t.endsWith('"')) ||
    (t.startsWith("'") && t.endsWith("'")) ||
    (t.startsWith("`") && t.endsWith("`"))
  ) {
    return t.slice(1, -1);
  }
  return t;
}

function readPromptFile(promptFile) {
  const p = path.resolve(promptFile);
  if (!fs.existsSync(p)) {
    throw new Error(`Prompt file does not exist: ${p}`);
  }
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
  const re = /^BEGIN_GPTWEB_FILE([^\r\n]*)\r?\n([\s\S]*?)^END_GPTWEB_FILE([^\r\n]*)(?:\r?\n|$)/gm;

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

async function findPromptBox(page, timeoutMs) {
  const candidates = [
    '[data-testid="prompt-textarea"]',
    '#prompt-textarea',
    'div[contenteditable="true"][id="prompt-textarea"]',
    '.ProseMirror[contenteditable="true"]',
    'textarea[placeholder*="Message"]',
    'textarea',
    'div[contenteditable="true"]',
    '[contenteditable="true"]',
  ];

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    for (const selector of candidates) {
      const loc = page.locator(selector).last();
      try {
        await loc.waitFor({ state: "visible", timeout: Math.min(2500, Math.max(500, deadline - Date.now())) });
        return loc;
      } catch (_) {}
    }
    await page.waitForTimeout(250);
  }
  throw new Error("Could not find ChatGPT prompt box.");
}

function isChatGptUrl(url) {
  return /^https?:\/\/(chatgpt\.com|chat\.openai\.com)(\/|$)/i.test(url || "");
}

async function listChatGptPages(browser) {
  const pages = [];
  for (const context of browser.contexts()) {
    for (const page of context.pages()) {
      const url = page.url();
      if (!isChatGptUrl(url)) continue;
      let title = "";
      try { title = await page.title(); } catch (_) {}
      pages.push({ page, url, title });
    }
  }
  return pages;
}

async function findExistingChatGptPage(browser, pageTimeoutMs, promptTimeoutMs) {
  const candidates = await listChatGptPages(browser);
  if (candidates.length === 0) {
    throw new Error(`No open ChatGPT tab found. Open https://chatgpt.com/ in Chrome running at ${CDP_URL}, then rerun.`);
  }

  for (const candidate of candidates.slice().reverse()) {
    const page = candidate.page;
    try {
      if (page.isClosed()) continue;
      await page.bringToFront();
      await page.waitForLoadState("domcontentloaded", { timeout: Math.min(10000, pageTimeoutMs) }).catch(() => {});
      await findPromptBox(page, Math.min(promptTimeoutMs, 30000));
      console.log(`Using existing ChatGPT tab: ${candidate.title || "(untitled)"}`);
      console.log(`URL: ${candidate.url}`);
      return page;
    } catch (_) {}
  }
  throw new Error("Found ChatGPT tab(s), but none had a visible prompt box. Make sure you are logged in and the tab is ready.");
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

    if (count > beforeCount) {
      current = await loc.nth(count - 1).innerText().catch(() => "");
    } else if (Date.now() > deadline - Math.min(30000, responseTimeoutMs / 4)) {
      current = await lastAssistantText(page);
    }

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

function setSystemClipboardText(text) {
  if (process.platform === "darwin") {
    const result = childProcess.spawnSync("/usr/bin/pbcopy", [], {
      input: text,
      encoding: "utf8",
      maxBuffer: 1024 * 1024,
    });
    if (result.status === 0) return true;
    throw new Error(`pbcopy failed: ${(result.stderr || result.error || "").toString().trim()}`);
  }

  const candidates = [
    ["wl-copy", []],
    ["xclip", ["-selection", "clipboard"]],
    ["xsel", ["--clipboard", "--input"]],
  ];
  for (const [cmd, argv] of candidates) {
    const result = childProcess.spawnSync(cmd, argv, {
      input: text,
      encoding: "utf8",
      maxBuffer: 1024 * 1024,
    });
    if (result.status === 0) return true;
  }
  return false;
}

async function grantClipboardPermissions(context) {
  for (const origin of ["https://chatgpt.com", "https://chat.openai.com"]) {
    try {
      await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin });
    } catch (_) {}
  }
}

async function setBrowserClipboardText(page, text) {
  await grantClipboardPermissions(page.context());
  return await page.evaluate(async (value) => {
    await navigator.clipboard.writeText(value);
    return true;
  }, text).catch(() => false);
}

async function promptBoxState(promptBox) {
  return await promptBox.evaluate((el) => {
    const text = "value" in el ? String(el.value || "") : String(el.innerText || el.textContent || "");
    return { length: text.length, prefix: text.slice(0, 80), suffix: text.slice(Math.max(0, text.length - 80)) };
  }).catch(() => ({ length: 0, prefix: "", suffix: "" }));
}

async function enabledSendButton(page) {
  const selectors = [
    'button[data-testid="send-button"]',
    'button[aria-label="Send prompt"]',
    'button[aria-label="Send message"]',
    'button[type="submit"]',
  ];

  for (const selector of selectors) {
    const loc = page.locator(selector).last();
    const count = await loc.count().catch(() => 0);
    if (count === 0) continue;
    const visible = await loc.isVisible().catch(() => false);
    if (!visible) continue;
    const disabled = await loc.isDisabled().catch(() => true);
    if (!disabled) return loc;
  }
  return null;
}

async function focusAndClearComposer(page, promptBox) {
  const mod = process.platform === "darwin" ? "Meta" : "Control";
  await promptBox.scrollIntoViewIfNeeded().catch(() => {});
  await promptBox.click({ timeout: 15000, force: true });
  await page.waitForTimeout(250);
  await page.keyboard.press(`${mod}+A`);
  await page.keyboard.press("Backspace");
  await page.waitForTimeout(300);
}

async function pastePlainTextPrompt(page, fullPrompt, promptTimeoutMs) {
  let promptBox = await findPromptBox(page, promptTimeoutMs);
  await focusAndClearComposer(page, promptBox);

  let clipboardOk = false;
  try {
    clipboardOk = setSystemClipboardText(fullPrompt);
  } catch (err) {
    console.log(`System clipboard setup failed: ${err.message}`);
  }
  if (!clipboardOk) clipboardOk = await setBrowserClipboardText(page, fullPrompt);
  if (!clipboardOk) throw new Error("Could not write prompt to system or browser clipboard.");

  const mod = process.platform === "darwin" ? "Meta" : "Control";
  console.log(`Pasting prompt as plain text with ${mod}+Shift+V (${fullPrompt.length} characters).`);
  await page.keyboard.press(`${mod}+Shift+V`);

  const deadline = Date.now() + promptTimeoutMs;
  let lastState = { length: 0 };
  let lastLog = 0;

  while (Date.now() < deadline) {
    promptBox = await findPromptBox(page, Math.min(5000, Math.max(1000, deadline - Date.now()))).catch(() => promptBox);
    lastState = await promptBoxState(promptBox);
    const send = await enabledSendButton(page);
    if (send) {
      console.log(`Prompt is ready to send; observed composer length=${lastState.length}.`);
      return send;
    }

    const now = Date.now();
    if (now - lastLog > 5000) {
      console.log(`Waiting for ChatGPT composer/send button after plain-text paste; composer length=${lastState.length}.`);
      lastLog = now;
    }
    await page.waitForTimeout(500);
  }

  throw new Error(`Plain-text paste did not make the send button ready; final observed composer length=${lastState.length}.`);
}

async function submitPrompt(page, fullPrompt, promptTimeoutMs) {
  const send = await pastePlainTextPrompt(page, fullPrompt, promptTimeoutMs);
  try {
    await send.click({ timeout: 15000 });
    return;
  } catch (err) {
    console.log(`Send button click failed (${err.message}); pressing Enter as fallback.`);
  }
  await page.keyboard.press("Enter");
}

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
        download: el.getAttribute("download") || "",
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
  if (browser && typeof browser.disconnect === "function") {
    await browser.disconnect().catch(() => {});
  } else if (browser) {
    await browser.close().catch(() => {});
  }
}

function buildFullPrompt(userPrompt, token) {
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

async function writeAnswerAndFiles(page, metaDir, outDir, promptFile, token, mode, extraManifest) {
  const answer = extraManifest.answer || "";
  const answerPath = path.join(metaDir, "answer.md");
  fs.writeFileSync(answerPath, answer || "", "utf8");

  const parsedFiles = parseFileBlocks(answer || "", token);
  const { written, skipped } = writeGeneratedFiles(parsedFiles, outDir);

  await dumpDebugState(page, metaDir);

  const manifest = {
    ok: written.length > 0,
    mode,
    promptFile: path.resolve(promptFile),
    outDir,
    metaDir,
    answerPath,
    token,
    files: written,
    skipped,
    parsedBlockCount: parsedFiles.length,
    ...extraManifest,
    note:
      written.length === 0
        ? "No files were written. Check .gpt-web-run/answer.md and .gpt-web-run/after-generation.png."
        : "Parsed GPTWEB_FILE blocks and wrote generated files directly under --out.",
  };
  delete manifest.answer;

  fs.writeFileSync(path.join(metaDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  console.log(JSON.stringify(manifest, null, 2));
  if (written.length === 0) process.exitCode = 2;
}

async function main() {
  const args = parseArgs(process.argv);
  const promptFile = requireArg(args, "prompt-file");
  const outDir = path.resolve(requireArg(args, "out"));
  const metaDir = path.join(outDir, ".gpt-web-run");
  const closeTab = boolArg(args, "close-tab");

  const connectTimeoutMs = intArg(args, "connect-timeout-ms", 120000);
  const pageTimeoutMs = intArg(args, "page-timeout-ms", 120000);
  const promptTimeoutMs = intArg(args, "prompt-timeout-ms", 120000);
  const responseTimeoutMs = intArg(args, "response-timeout-ms", 1200000);

  mkdirp(outDir);
  mkdirp(metaDir);

  const userPrompt = readPromptFile(promptFile);
  const token = randomToken();
  const fullPrompt = buildFullPrompt(userPrompt, token);

  const browser = await chromium.connectOverCDP(CDP_URL, { timeout: connectTimeoutMs });
  let page = null;

  try {
    await denyMicIfPossible(browser);
    const context = browser.contexts()[0];
    if (!context) throw new Error(`No existing Chrome context found at ${CDP_URL}`);
    context.setDefaultTimeout(pageTimeoutMs);

    page = await context.newPage();
    await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
    await page.goto("https://chatgpt.com/", { waitUntil: "domcontentloaded", timeout: pageTimeoutMs });
    await page.waitForLoadState("domcontentloaded", { timeout: pageTimeoutMs }).catch(() => {});
    await page.waitForTimeout(2500);

    await findPromptBox(page, promptTimeoutMs);
    const beforeCount = await assistantCount(page);
    await submitPrompt(page, fullPrompt, promptTimeoutMs);

    console.log("Prompt submitted in new ChatGPT tab. Waiting for response to settle...");
    const answer = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs);

    await writeAnswerAndFiles(page, metaDir, outDir, promptFile, token, "new-chatgpt-tab-gptweb-file-parser", {
      answer,
      keptChatGptTabOpen: !closeTab,
      chatGptTabUrl: page.url(),
    });
  } finally {
    if (closeTab && page) await page.close().catch(() => {});
    await disconnectBrowser(browser);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
