---
name: playwright-cli
description: Drive a real browser to inspect and interact with a deployed web page for functional verification. Use when you need to open a URL, read the DOM, click/type, or screenshot a page.
---

# playwright-cli

You have a Playwright browser available through the **`mcp__playwright__browser_*`**
MCP tools (the `playwright` MCP server is already connected — do NOT try to
install playwright, run `npx playwright`, or write your own Node script). Use the
MCP tools directly.

## Core tools

| Goal | Tool | Notes |
|------|------|-------|
| Open a page | `mcp__playwright__browser_navigate` | pass `{ "url": "<URL>" }` |
| Read the DOM (preferred) | `mcp__playwright__browser_snapshot` | returns an accessibility tree with `ref` ids for every element |
| Click | `mcp__playwright__browser_click` | pass the element `ref` from the latest snapshot |
| Type text | `mcp__playwright__browser_type` | `{ "ref": ..., "text": ..., "submit": true|false }` |
| Press a key | `mcp__playwright__browser_press_key` | e.g. `Enter` |
| Run JS / read state | `mcp__playwright__browser_evaluate` | `{ "function": "() => <expr>" }` — use to read computed values, element text, counts |
| Screenshot (visual checks) | `mcp__playwright__browser_take_screenshot` | full page image |
| Wait | `mcp__playwright__browser_wait_for` | wait for text to appear/disappear |

## How to verify a requirement

1. `browser_navigate` to the URL once at the start.
2. Take a `browser_snapshot` to see the current DOM and get element `ref`s.
3. For interaction requirements: `browser_click` / `browser_type` the relevant
   `ref`s, then snapshot again (or `browser_evaluate`) to confirm the result
   actually changed (e.g. the display shows the expected number).
4. For visual requirements: `browser_take_screenshot` and look.
5. Decide **yes** or **no** for each requirement.

## Rules

- If an element can't be found or the page errors → **no**.
- Do NOT fix the page, mock data, or inject scripts to make a check pass — only
  observe.
- Prefer `browser_snapshot` / `browser_evaluate` for functional/DOM checks;
  use `browser_take_screenshot` for visual ones.
- When unsure, lean **no**.
