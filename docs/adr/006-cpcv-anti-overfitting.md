# ADR 006: Combinatorial Purged Cross-Validation for Anti-Overfitting

**Status:** Accepted  
**Context:** Walk-forward analysis was insufficient for preventing overfitting; strategies passed validation but failed on unseen data.  
**Decision:** Replace walk-forward (Gate 6) with Combinatorial Purged Cross-Validation (CPCV) using CPCVSplitter + CPCVValidator. Implements blind parameter search on holdout data and synthetic sanity checks.  
**Consequences:** 11-gate deployment pipeline, stronger overfitting detection. 37 dedicated tests for CPCV logic.  
**Date:** 2026-06-03
