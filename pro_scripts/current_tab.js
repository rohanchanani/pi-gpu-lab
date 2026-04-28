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
  if (args[name] === undefined) return defaultValue;
  const n = Number(args[name]);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`Invalid integer argument --${name}: ${args[name]}`);
  }
  return Math.floor(n);
}

function requireArg(args, name) {
  if (!args[name]) throw new Error(`Missing required argument: --${name}`);
  return args[name];
}

function mkdirp(p) {
  fs.mkdirSync(p, { recursive: true });
}

function escapeRegExp(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function stripOuterQuotes(s) {
  const t = String(s || "").trim();
  if ((t.startsWith('"') && t.endsWith('"')) ||
      (t.startsWith("'") && t.endsWith("'")) ||
      (t.startsWith("`") && t.endsWith("`"))) {
    return t.slice(1, -1);
  }
  return t;
}

function readPromptFile(promptFile) {
  const p = path.resolve(promptFile);
  if (!fs.existsSync(p)) throw new Error(`Prompt file does not exist: ${p}`);
  return fs.readFileSync(p, "utf8");
}

function parseKv(header, key) {
  const re = new RegExp(`(?:^|\\s)${escapeRegExp(key)}=(?:\"([^\"]*)\"|'([^']*)'|(\\S+))`);
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
  h = h.replace(/(?:^|\s)token=(?:\"[^\"]*\"|'[^']*'|\S+)/g, "").trim();
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

    // The god prompt requires token=abc123.  The wrapper also uses abc123.
    // Accept missing tokens and abc123; reject other explicit tokens.
    if (expectedToken) {
      if (headerToken && headerToken !== expectedToken) continue;
      if (footerToken && footerToken !== expectedToken) continue;
    }

    try {
      files.push({ relPath: parseHeaderPath(header), content, header });
    } catch (err) {
      files.push({ error: err.message, header, skipped: true });
    }
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

async function findPromptBox(page, timeout = 30000) {
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

  const perCandidate = Math.max(1000, Math.floor(timeout / candidates.length));
  for (const selector of candidates) {
    const loc = page.locator(selector).last();
    try {
      await loc.waitFor({ state: "visible", timeout: perCandidate });
      return loc;
    } catch (_) {}
  }

  throw new Error("Could not find ChatGPT prompt box.");
}

async function focusPromptBox(page, promptBox) {
  await page.bringToFront().catch(() => {});
  await promptBox.scrollIntoViewIfNeeded().catch(() => {});
  await promptBox.click({ timeout: 15000, force: true }).catch(async () => {
    await page.mouse.click(700, 930);
  });
  await page.waitForTimeout(250);

  await page.evaluate(() => {
    const selectors = [
      '[data-testid="prompt-textarea"]',
      '#prompt-textarea',
      '.ProseMirror[contenteditable="true"]',
      'textarea[placeholder*="Message"]',
      'textarea',
      'div[contenteditable="true"]',
      '[contenteditable="true"]',
    ];
    for (const selector of selectors) {
      const els = Array.from(document.querySelectorAll(selector));
      for (const el of els.reverse()) {
        const rect = el.getBoundingClientRect();
        const visible = rect.width > 0 && rect.height > 0;
        if (!visible) continue;
        el.focus();
        if (el.isContentEditable) {
          try {
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(el);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
          } catch (_) {}
        }
        return true;
      }
    }
    return false;
  }).catch(() => {});
  await page.waitForTimeout(250);
}

async function clearComposer(page) {
  const mod = process.platform === "darwin" ? "Meta" : "Control";
  await page.keyboard.press(`${mod}+A`).catch(() => {});
  await page.keyboard.press("Backspace").catch(() => {});
  await page.waitForTimeout(300);
}

function putSystemClipboard(text) {
  if (process.platform === "darwin") {
    const proc = childProcess.spawnSync("/usr/bin/pbcopy", [], {
      input: text,
      encoding: "utf8",
      maxBuffer: Math.max(1024 * 1024, Buffer.byteLength(text, "utf8") + 1024),
    });
    if (proc.status !== 0) {
      throw new Error(`pbcopy failed: ${(proc.stderr || "").toString()}`);
    }
    return;
  }

  // Linux fallback: xclip or wl-copy if installed.
  for (const cmd of ["wl-copy", "xclip"]) {
    const exists = childProcess.spawnSync("/bin/sh", ["-lc", `command -v ${cmd}`], { encoding: "utf8" });
    if (exists.status !== 0) continue;
    const args = cmd === "xclip" ? ["-selection", "clipboard"] : [];
    const proc = childProcess.spawnSync(cmd, args, { input: text, encoding: "utf8" });
    if (proc.status === 0) return;
  }

  throw new Error("No supported system clipboard command found; install wl-copy or xclip.");
}

function runOsa(scriptLines) {
  const args = [];
  for (const line of scriptLines) args.push("-e", line);
  return childProcess.spawnSync("/usr/bin/osascript", args, {
    encoding: "utf8",
    timeout: 10000,
  });
}

async function nativePlainPaste(page, fullPrompt) {
  putSystemClipboard(fullPrompt);

  if (process.platform === "darwin") {
    const appCandidates = [];
    if (process.env.GPT_WEB_BROWSER_APP) appCandidates.push(process.env.GPT_WEB_BROWSER_APP);
    appCandidates.push("Google Chrome", "Chromium", "Google Chrome Canary");

    for (const app of appCandidates) {
      const proc = runOsa([
        `tell application ${JSON.stringify(app)} to activate`,
        "delay 0.25",
        'tell application "System Events" to keystroke "v" using {command down, shift down}',
      ]);
      if (proc.status === 0) return true;
      console.log(`Native paste attempt via ${app} failed: ${(proc.stderr || "").trim()}`);
    }

    const proc = runOsa([
      'tell application "System Events" to keystroke "v" using {command down, shift down}',
    ]);
    if (proc.status === 0) return true;
    console.log(`Native paste attempt without app activation failed: ${(proc.stderr || "").trim()}`);
  }

  // Playwright fallback.  Keep this as plain-text paste (Shift+Paste), not normal paste.
  const mod = process.platform === "darwin" ? "Meta" : "Control";
  await page.keyboard.down(mod);
  await page.keyboard.down("Shift");
  await page.keyboard.press("V");
  await page.keyboard.up("Shift");
  await page.keyboard.up(mod);
  return true;
}

async function composerTextLength(page) {
  return await page.evaluate(() => {
    const selectors = [
      '[data-testid="prompt-textarea"]',
      '#prompt-textarea',
      '.ProseMirror[contenteditable="true"]',
      'textarea[placeholder*="Message"]',
      'textarea',
      'div[contenteditable="true"]',
      '[contenteditable="true"]',
    ];
    let best = 0;
    for (const selector of selectors) {
      for (const el of Array.from(document.querySelectorAll(selector))) {
        const rect = el.getBoundingClientRect();
        const visible = rect.width > 0 && rect.height > 0;
        if (!visible) continue;
        const value = el.value || el.innerText || el.textContent || "";
        if (value.length > best) best = value.length;
      }
    }
    return best;
  }).catch(() => 0);
}

async function findEnabledSendButton(page) {
  const selectors = [
    'button[data-testid="send-button"]',
    'button[aria-label="Send prompt"]',
    'button[aria-label="Send message"]',
    'button[aria-label="Send"]',
  ];

  for (const selector of selectors) {
    const loc = page.locator(selector);
    const count = await loc.count().catch(() => 0);
    for (let i = count - 1; i >= 0; i--) {
      const button = loc.nth(i);
      const visible = await button.isVisible().catch(() => false);
      if (!visible) continue;
      const disabled = await button.isDisabled().catch(() => true);
      if (!disabled) return button;
    }
  }
  return null;
}

async function waitForPromptReady(page, timeoutMs, expectedLength) {
  const start = Date.now();
  let lastLog = 0;
  while (Date.now() - start < timeoutMs) {
    const send = await findEnabledSendButton(page);
    const len = await composerTextLength(page);
    if (send) {
      console.log(`Prompt is ready to send; observed composer length=${len}.`);
      return send;
    }

    if (Date.now() - lastLog > 5000) {
      console.log(
        `Waiting for ChatGPT composer/send button after plain-text paste; composer length=${len}, expected approx=${expectedLength}.`
      );
      lastLog = Date.now();
    }
    await page.waitForTimeout(1000);
  }
  throw new Error("Plain-text paste did not enable ChatGPT send button before timeout.");
}

async function submitPrompt(page, promptBox, fullPrompt, promptTimeoutMs) {
  console.log(`Pasting prompt as plain text with native Cmd+Shift+V (${fullPrompt.length} characters).`);
  await focusPromptBox(page, promptBox);
  await clearComposer(page);
  await focusPromptBox(page, promptBox);
  await nativePlainPaste(page, fullPrompt);

  const sendButton = await waitForPromptReady(page, promptTimeoutMs, fullPrompt.length);
  await sendButton.click({ timeout: 15000 });
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
  const start = Date.now();
  while (Date.now() - start < responseTimeoutMs) {
    await page.waitForTimeout(1000);
    const loc = page.locator('[data-message-author-role="assistant"]');
    const count = await loc.count().catch(() => 0);
    let current = "";
    if (count > beforeCount) {
      current = await loc.nth(count - 1).innerText().catch(() => "");
    } else if (Date.now() - start > 20000) {
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
  mkdirp(metaDir);
  await page.screenshot({ path: path.join(metaDir, "after-generation.png"), fullPage: true }).catch(() => {});
  const candidates = await page.evaluate(() => {
    const els = [...document.querySelectorAll("a, button, textarea, div[contenteditable='true'], [data-testid]")];
    return els.map((el, i) => {
      const rect = el.getBoundingClientRect();
      return {
        i,
        tag: el.tagName.toLowerCase(),
        text: (el.innerText || el.textContent || "").trim().slice(0, 250),
        aria: el.getAttribute("aria-label") || "",
        id: el.getAttribute("id") || "",
        role: el.getAttribute("role") || "",
        testid: el.getAttribute("data-testid") || "",
        className: String(el.getAttribute("class") || "").slice(0, 250),
        disabled: !!el.disabled,
        visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
        rect: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) },
      };
    });
  }).catch(() => []);
  fs.writeFileSync(path.join(metaDir, "click-candidates.json"), JSON.stringify(candidates, null, 2), "utf8");
}

async function findExistingChatGptPage(browser, pageTimeoutMs) {
  const candidates = await listChatGptPages(browser);
  if (candidates.length === 0) {
    throw new Error(`No open ChatGPT tab found. Open https://chatgpt.com/ in Chrome running at ${CDP_URL}, then rerun.`);
  }
  for (const candidate of candidates.slice().reverse()) {
    const page = candidate.page;
    try {
      if (page.isClosed()) continue;
      await page.bringToFront();
      await page.waitForLoadState("domcontentloaded", { timeout: Math.min(pageTimeoutMs, 10000) }).catch(() => {});
      await findPromptBox(page, Math.min(pageTimeoutMs, 30000));
      console.log(`Using existing ChatGPT tab: ${candidate.title || "(untitled)"}`);
      console.log(`URL: ${candidate.url}`);
      return page;
    } catch (_) {}
  }
  throw new Error("Found ChatGPT tab(s), but none had a visible prompt box. Make sure you are logged in and the tab is ready.");
}

async function disconnectBrowser(browser) {
  if (browser && typeof browser.disconnect === "function") await browser.disconnect().catch(() => {});
  else if (browser) await browser.close().catch(() => {});
}

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

async function run(mode) {
  const args = parseArgs(process.argv);
  const promptFile = requireArg(args, "prompt-file");
  const outDir = path.resolve(requireArg(args, "out"));
  const metaDir = path.join(outDir, ".gpt-web-run");
  const closeTab = boolArg(args, "close-tab");

  const connectTimeoutMs = intArg(args, "connect-timeout-ms", 120000);
  const pageTimeoutMs = intArg(args, "page-timeout-ms", 120000);
  const promptTimeoutMs = intArg(args, "prompt-timeout-ms", 120000);
  const responseTimeoutMs = intArg(args, "response-timeout-ms", 1200000);
  const token = args.token || "abc123";

  mkdirp(outDir);
  mkdirp(metaDir);

  const userPrompt = readPromptFile(promptFile);
  const fullPrompt = buildWrappedPrompt(userPrompt, token);

  const browser = await chromium.connectOverCDP(CDP_URL, { timeout: connectTimeoutMs });
  let page = null;

  try {
    await denyMicIfPossible(browser);
    let context = browser.contexts()[0];
    if (!context) throw new Error(`No existing Chrome context found at ${CDP_URL}`);
    context.setDefaultTimeout(pageTimeoutMs);

    if (mode === "new") {
      page = await context.newPage();
      await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
      await page.goto("https://chatgpt.com/", { waitUntil: "domcontentloaded", timeout: pageTimeoutMs });
      await page.waitForTimeout(4000);
      console.log("Opened new ChatGPT tab.");
    } else {
      page = await findExistingChatGptPage(browser, pageTimeoutMs);
    }

    await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
    await page.bringToFront();

    const promptBox = await findPromptBox(page, pageTimeoutMs);
    const beforeCount = await assistantCount(page);

    await submitPrompt(page, promptBox, fullPrompt, promptTimeoutMs);

    console.log(mode === "new"
      ? "Prompt submitted in new ChatGPT tab. Waiting for response to settle..."
      : "Prompt submitted into existing ChatGPT tab. Waiting for response to settle...");

    const answer = await waitForNewAssistantToSettle(page, beforeCount, responseTimeoutMs);
    const answerPath = path.join(metaDir, "answer.md");
    fs.writeFileSync(answerPath, answer || "", "utf8");

    const parsedFiles = parseFileBlocks(answer || "", token);
    const { written, skipped } = writeGeneratedFiles(parsedFiles, outDir);

    await dumpDebugState(page, metaDir);

    const manifest = {
      ok: written.length > 0,
      mode: mode === "new" ? "new-chatgpt-tab-gptweb-file-parser" : "existing-chatgpt-tab-gptweb-file-parser",
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
      note: written.length === 0
        ? "No files were written. Check .gpt-web-run/answer.md and .gpt-web-run/after-generation.png."
        : "Parsed GPTWEB_FILE blocks and wrote generated files directly under --out.",
    };

    fs.writeFileSync(path.join(metaDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
    console.log(JSON.stringify(manifest, null, 2));
    if (written.length === 0) process.exitCode = 2;
  } catch (err) {
    if (page) await dumpDebugState(page, metaDir).catch(() => {});
    throw err;
  } finally {
    if (closeTab && page) await page.close().catch(() => {});
    await disconnectBrowser(browser);
  }
}

const mode = process.env.GPT_WEB_MODE || (path.basename(process.argv[1]).startsWith("new_") ? "new" : "current");
run(mode).catch((err) => {
  console.error(err);
  process.exit(1);
});
