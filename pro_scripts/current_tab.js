#!/usr/bin/env node

const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

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

  if (!p) {
    throw new Error("Empty GPTWEB_FILE path");
  }

  if (p.startsWith("/")) {
    throw new Error(`Refusing absolute path: ${p}`);
  }

  const parts = p.split("/").filter((part) => part.length > 0 && part !== ".");

  if (parts.some((part) => part === "..")) {
    throw new Error(`Refusing path with '..': ${p}`);
  }

  const normalized = parts.join("/");

  if (!normalized) {
    throw new Error(`Invalid GPTWEB_FILE path: ${p}`);
  }

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

  const re =
    /^BEGIN_GPTWEB_FILE([^\r\n]*)\r?\n([\s\S]*?)^END_GPTWEB_FILE([^\r\n]*)(?:\r?\n|$)/gm;

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
      files.push({
        error: err.message,
        header,
        skipped: true,
      });
      continue;
    }

    files.push({
      relPath,
      content,
      header,
    });
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
      skipped.push({
        relPath: f.relPath,
        error: err.message,
      });
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
      try {
        title = await page.title();
      } catch (_) {}

      pages.push({ page, url, title });
    }
  }

  return pages;
}

async function findPromptBox(page, timeout = 10000) {
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

  for (const selector of candidates) {
    const loc = page.locator(selector).last();
    try {
      await loc.waitFor({ state: "visible", timeout });
      return loc;
    } catch (_) {}
  }

  throw new Error("Could not find ChatGPT prompt box on the selected tab.");
}

async function findExistingChatGptPage(browser) {
  const candidates = await listChatGptPages(browser);

  if (candidates.length === 0) {
    throw new Error(
      `No open ChatGPT tab found. Open https://chatgpt.com/ in Chrome running at ${CDP_URL}, then rerun.`
    );
  }

  for (const candidate of candidates.slice().reverse()) {
    const page = candidate.page;

    try {
      if (page.isClosed()) continue;

      await page.bringToFront();
      await page.waitForLoadState("domcontentloaded", { timeout: 5000 }).catch(() => {});
      await findPromptBox(page, 5000);

      console.log(`Using existing ChatGPT tab: ${candidate.title || "(untitled)"}`);
      console.log(`URL: ${candidate.url}`);

      return page;
    } catch (_) {}
  }

  throw new Error(
    "Found ChatGPT tab(s), but none had a visible prompt box. Make sure you are logged in and the tab is ready."
  );
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

async function waitForNewAssistantToSettle(page, beforeCount) {
  let previous = "";
  let stableCount = 0;
  let lastNonEmpty = "";

  for (let i = 0; i < 240; i++) {
    await page.waitForTimeout(1000);

    const loc = page.locator('[data-message-author-role="assistant"]');
    const count = await loc.count().catch(() => 0);

    let current = "";

    if (count > beforeCount) {
      current = await loc.nth(count - 1).innerText().catch(() => "");
    } else if (i > 20) {
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

async function fillPromptBox(page, promptBox, fullPrompt) {
  try {
    await promptBox.fill(fullPrompt, { timeout: 5000 });
    return;
  } catch (_) {}

  await promptBox.click();

  const mod = process.platform === "darwin" ? "Meta" : "Control";
  await page.keyboard.press(`${mod}+A`);
  await page.keyboard.press("Backspace");
  await page.keyboard.insertText(fullPrompt);
}

async function submitPrompt(page, promptBox, fullPrompt) {
  await fillPromptBox(page, promptBox, fullPrompt);
  await page.waitForTimeout(700);

  const sendSelectors = [
    'button[data-testid="send-button"]',
    'button[aria-label="Send prompt"]',
    'button[aria-label="Send message"]',
  ];

  for (const selector of sendSelectors) {
    const loc = page.locator(selector).last();
    const count = await loc.count().catch(() => 0);
    if (count === 0) continue;

    try {
      await loc.waitFor({ state: "visible", timeout: 3000 });

      for (let i = 0; i < 10; i++) {
        const disabled = await loc.isDisabled().catch(() => true);
        if (!disabled) {
          await loc.click({ timeout: 5000 });
          return;
        }
        await page.waitForTimeout(500);
      }
    } catch (_) {}
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
  await page
    .screenshot({
      path: path.join(metaDir, "after-generation.png"),
      fullPage: true,
    })
    .catch(() => {});

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

  fs.writeFileSync(
    path.join(metaDir, "click-candidates.json"),
    JSON.stringify(candidates, null, 2),
    "utf8"
  );
}

async function disconnectBrowser(browser) {
  if (browser && typeof browser.disconnect === "function") {
    await browser.disconnect().catch(() => {});
  } else if (browser) {
    await browser.close().catch(() => {});
  }
}

async function main() {
  const args = parseArgs(process.argv);

  const promptFile = requireArg(args, "prompt-file");
  const outDir = path.resolve(requireArg(args, "out"));
  const metaDir = path.join(outDir, ".gpt-web-run");

  mkdirp(outDir);
  mkdirp(metaDir);

  const userPrompt = readPromptFile(promptFile);
  const token = randomToken();

  const fullPrompt = `
Use the fastest available model. Do not use a thinking/reasoning mode.

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

  const browser = await chromium.connectOverCDP(CDP_URL);

  try {
    await denyMicIfPossible(browser);

    const page = await findExistingChatGptPage(browser);
    const context = page.context();
    context.setDefaultTimeout(15000);

    await page.setViewportSize({ width: 1400, height: 1000 }).catch(() => {});
    await page.bringToFront();

    const promptBox = await findPromptBox(page);
    const beforeCount = await assistantCount(page);

    await submitPrompt(page, promptBox, fullPrompt);

    console.log("Prompt submitted into existing ChatGPT tab. Waiting for response to settle...");
    const answer = await waitForNewAssistantToSettle(page, beforeCount);

    const answerPath = path.join(metaDir, "answer.md");
    fs.writeFileSync(answerPath, answer || "", "utf8");

    const parsedFiles = parseFileBlocks(answer || "", token);
    const { written, skipped } = writeGeneratedFiles(parsedFiles, outDir);

    await dumpDebugState(page, metaDir);

    const manifest = {
      ok: written.length > 0,
      mode: "existing-chatgpt-tab-gptweb-file-parser",
      promptFile: path.resolve(promptFile),
      outDir,
      metaDir,
      answerPath,
      token,
      files: written,
      skipped,
      parsedBlockCount: parsedFiles.length,
      note:
        written.length === 0
          ? "No files were written. Check .gpt-web-run/answer.md and .gpt-web-run/after-generation.png."
          : "Parsed GPTWEB_FILE blocks and wrote generated files directly under --out.",
    };

    fs.writeFileSync(
      path.join(metaDir, "manifest.json"),
      JSON.stringify(manifest, null, 2),
      "utf8"
    );

    console.log(JSON.stringify(manifest, null, 2));

    if (written.length === 0) {
      process.exitCode = 2;
    }
  } finally {
    await disconnectBrowser(browser);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
