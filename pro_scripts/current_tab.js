#!/usr/bin/env node
const { run } = require("./gpt_web_driver");
run("current", process.argv).catch((err) => {
  console.error(err);
  process.exit(1);
});
