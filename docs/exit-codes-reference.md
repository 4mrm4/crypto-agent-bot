# Exit Codes Reference

Exit codes encountered during development and how to diagnose/fix them.

## Exit Code 1 — General Error

**What it means:** The command or tool exited with an unspecified error. Python uses exit code 1 for most runtime exceptions.

### Scenario A: Silent failure on Windows Bash (most common trap)

**Symptom:** You run `python -c "..."` or `python script.py` from bash on Windows and get:
```
Error: Exit code 1
```
with **no output at all** — no traceback, no error message, nothing.

**Root cause:** On Windows, Python's stderr uses the Windows console API, not the Unix stdout/stderr file descriptors. Bash on Windows (msys2/Git Bash) uses `2>&1` which is a Unix shell redirect and does NOT properly capture Windows Python's stderr output. The Python process crashes, writes the traceback to the Windows console handle, exits with code 1, and the bash shell sees exit code 1 but never saw the error text.

**Diagnosis:**
1. Write a wrapper script that redirects Python's output at the Python level (not shell level):
   ```python
   # redirect_stderr.py
   import sys, traceback
   sys.stdout = open(r'C:\Temp\debug_out.txt', 'w', encoding='utf-8')
   sys.stderr = sys.stdout
   try:
       # your code here
       exec(open('target_file.py').read())
   except Exception as e:
       traceback.print_exc()
   sys.stdout.close()
   ```
2. Run the wrapper: `python C:\path\to\redirect_stderr.py`
3. Read the output file: `cat /c/Temp/debug_out.txt`

**Fix:** Once you see the actual error in the output file, fix the underlying code issue (usually a `SyntaxError` or `ImportError`).

### Scenario B: ast.parse() catches SyntaxError

**Symptom:** `ast.parse(open('file.py').read())` returns exit code 1.

**Root cause:** The file has a genuine syntax error (missing parenthesis, unclosed string, invalid syntax).

**Diagnosis:** Use the wrapper script approach above to see the exact `SyntaxError` with line number and error text.

### Scenario C: ImportError / NameError at module level

**Symptom:** `import mymodule` returns exit code 1.

**Root cause:** A module-level import fails (e.g., `from config import settings` when `config` isn't in `sys.path`) or a `NameError` occurs during module-level code execution.

**Diagnosis:** Try importing with proper `sys.path` setup:
```python
import sys
sys.path.insert(0, r'C:\path\to\project')
import mymodule
```

---

## Exit Code 2 — Python command syntax error / file not found

**Symptom:** Running `python something.py` gives exit code 2 and "Error: Exit code 2".

**What it means:**
- The `.py` file doesn't exist at the path you specified
- OR the file itself has a Python syntax error (not a runtime error — the interpreter can't parse the file at all)
- OR you're passing invalid flags to python

**Diagnosis:**
1. Verify the file exists: `ls -la /path/to/file.py`
2. Check file is not empty: `wc -l /path/to/file.py`
3. Try running with explicit `-c` to verify python works: `python -c "print('hello')"`

**Fix:**
- If file doesn't exist: correct the path
- If file has syntax error: use the stderr capture technique from Exit Code 1, Scenario A

---

## Exit Code 10048 — Port conflict (WSAEADDRINUSE)

**Symptom:**
```
[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8765):
address already in use
```

**What it means:** Another process is already listening on port 8765. This typically means a previous instance of the server is still running.

**Diagnosis:**
```bash
# Check what's on port 8765
netstat -ano | grep 8765

# Or with PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8765).OwningProcess
```

**Fix:**
```bash
# Kill the process on port 8765 (replace PID with actual)
taskkill /F /PID <PID>

# Or kill all python processes (if safe)
taskkill /F /IM python.exe
```

Note: This is harmless if intentional — just means you started a second instance.

---

## Exit Code 137 — Out of memory (SIGKILL)

**Symptom:** Process suddenly disappears with exit code 137.

**What it means:** The OS killed the process (out of memory, or container/cgroup limit exceeded). Common when loading large datasets or running many parallel backtests.

**Diagnosis:** Check system memory usage and the size of data being processed.

**Fix:** Reduce batch size, add memory limits, or use chunked processing.

---

## Exit Code 139 — Segmentation fault (SIGSEGV)

**Symptom:** Process crashes with exit code 139.

**What it means:** The Python process tried to access memory it shouldn't. Common with native extensions (TA-Lib, numpy, pandas) when they receive unexpected data types.

**Diagnosis:** Check for dtype mismatches — e.g., TA-Lib functions receiving string dtypes from PyArrow-backed DataFrames.

**Fix:** Add explicit `.astype(float)` coercion before passing data to native libraries.

---

## Exit Code 202 — HTTP Accepted (non-fatal, external API)

**Symptom:** DuckDuckGo search returns HTTP 202.

**What it means:** The request was accepted but no results are available yet (or rate-limited). NOT a code bug.

**Fix:** Configure a proper search API key (Tavily) for the researcher agent.

---

## General Troubleshooting Flow

When you get a silent/empty exit code on Windows:

1. **Is it exit code 2?** → File not found or Python syntax error in the script itself
2. **Is it exit code 1 with no output?** → Use the Python-level stderr redirect wrapper
3. **Is it a known Windows bash issue?** → Prefer writing `.py` script files over `python -c` inline commands
4. **Can you reproduce in PowerShell?** → PowerShell captures stderr more reliably:
   ```powershell
   python -c "import ast; ast.parse(open('file.py').read())" 2>&1
   ```
