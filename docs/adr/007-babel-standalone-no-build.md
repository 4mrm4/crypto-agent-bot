# ADR 007: Web UI with Babel Standalone (No Build Step)

**Status:** Accepted  
**Context:** Needed interactive web dashboard for monitoring and goal submission without adding a Node.js build pipeline or increasing deployment complexity.  
**Decision:** Use React with Babel standalone for in-browser JSX transformation. No webpack/vite. Server serves static HTML + JS.  
**Consequences:** Zero build step simplifies deployment. JSX in `<script type="text/babel">` tags. Trailing whitespace and unclosed function blocks can break Babel parsing; validated via @babel/parser binary search.  
**Date:** 2026-05-30
