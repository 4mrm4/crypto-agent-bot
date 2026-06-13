# ADR 010: Kraken Pro as Primary Execution Exchange

**Status:** Accepted  
**Context:** User switched from Binance as primary exchange for live trade execution due to regulatory/access considerations.  
**Decision:** Configure Kraken Pro as default execution exchange. .env + code defaults updated. CCXT unified API abstracts exchange differences.  
**Consequences:** Kraken-specific rate limits and API quirks. Paper trading mode still abstracts exchange entirely for testing.  
**Date:** 2026-06-04
