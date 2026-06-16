---
name: ui-babel-standalone-blank-page
description: Web UI blank page fix — Babel standalone CDN and auto-transform issues
type: project
---

# Web UI renders blank — Babel Standalone auto-transform fails

## Symptom
`python main.py --autonomous --ui` starts the server correctly, HTML is served, React/ReactDOM/Babel load from CDN, but the root div is empty with no JS errors.

## Root cause
Babel standalone's automatic `transformScriptTags()` detection does not fire reliably. The `<script type="text/babel">` block containing the React JSX app is never compiled/executed.

## Fix applied (2026-06-16)
1. **Removed `crossorigin`** from Babel CDN script tag — this attribute forces CORS mode which can interfere with Babel's DOMContentLoaded handler
2. **Pinned Babel to 7.23.0**: `https://unpkg.com/@babel/standalone@7.23.0/babel.min.js`
3. **Added fallback**: inline script after the text/babel block that checks if root div is empty after 500ms and manually calls `Babel.transformScriptTags()`

**How to apply:** When the UI is blank with no console errors, check that Babel's auto-transform is working. The fallback script handles this.
