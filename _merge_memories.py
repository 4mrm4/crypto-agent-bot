"""Merge 3 memory sources into a single workspace instructions file."""
import json, os, re

DUMP = r"C:\Trading-bot\crypto_agent_bot\memory_dump.json"
OUT  = r"C:\Trading-bot\crypto_agent_bot\.openclaude\instructions\trading-bot.instructions.md"
REPO_ROOT = r"C:\Trading-bot\crypto_agent_bot"

with open(DUMP, encoding="utf-8") as f:
    data = json.load(f)

bot    = data["bot"]["files"]
crypto = data["crypto"]["files"]
user   = data["user"]["files"]

# --- helpers ---
def norm_key(k):
    """Return a canonical key like 'feedback/foo.md'."""
    return k.replace("\\", "/").lstrip("/")

def get_text(src, *possible_keys):
    """Return first matching text from src dict."""
    for k in possible_keys:
        v = src.get(k)
        if v:
            return v
        v = src.get(k.replace("/", "\\"))
        if v:
            return v
    return ""

def strip_frontmatter(text):
    return re.sub(r"^---[\s\S]*?---\n*", "", text).strip()

# Normalize all keys
bot    = {norm_key(k): v for k, v in bot.items()}
crypto = {norm_key(k): v for k, v in crypto.items()}
user   = {norm_key(k): v for k, v in user.items()}

lines = []
def add(t):
    lines.append(t)

# ===== HEADER =====
add("# Crypto Agent Bot \u2014 Workspace Instructions")
add("")
add("> Merged from 3 OpenClaude memory sources by memory-merger skill")
add("> Sources: `C--Trading-bot/memory`, `C--Trading-bot-crypto-agent-bot`, `C--Users-Amral/memory`")
add("> Generated: 2026-06-14")
add("")
add("---")
add("")

# ===== CHUNK FUNCTIONS =====
def chunk_user_profile():
    add("## User Profile")
    add("")

    # Merge 3 sources
    a = strip_frontmatter(get_text(bot, "user/amral-profile.md"))
    c = strip_frontmatter(get_text(crypto, "user-profile.md"))
    u = strip_frontmatter(get_text(user, "user-profile.md"))

    add("### From C--Trading-bot/user/amral-profile.md")
    add("")
    add(a or "_empty_")
    add("")
    add("### From crypto-agent-bot/user-profile.md")
    add("")
    add(c or "_empty_")
    add("")
    add("### From C--Users-Amral/user-profile.md")
    add("")
    add(u or "_empty_")
    add("")

def chunk_feedback():
    add("## Feedback & Behavioral Rules")
    add("")

    # Collect all feedback files from bot
    keys = sorted(k for k in bot if k.startswith("feedback/"))
    for key in keys:
        name = os.path.splitext(os.path.basename(key))[0]
        content = strip_frontmatter(bot[key])
        if not content:
            continue
        add(f"### {name}")
        add("")
        add(content)
        add("")

    # Add crypto feedback-style files not already covered
    crypto_feedback_keys = [
        "avoid-unsolicited-fixes.md",
        "prefer-deletion-over-abstraction.md",
        "prefer-deterministic-over-llm-fixes.md",
        "prefer-test-first-lazy-import-conversion.md",
        "workflow-protocol-preference.md",
        "event-callback-property-fix.md",
        "powershell-syntax.md",
    ]
    for k in crypto_feedback_keys:
        name = os.path.splitext(k)[0]
        content = strip_frontmatter(get_text(crypto, k, f"feedback/{k}"))
        if not content:
            continue
        # Check if bot already covered it
        bot_key = f"feedback/{k}"
        if bot_key in bot:
            continue
        add(f"### {name} (from crypto-agent-bot)")
        add("")
        add(content)
        add("")

    # User feedback files
    user_feedback = ["feedback-autonomous-execution.md", "feedback-stop-asking-after-long-setup.md"]
    for k in user_feedback:
        name = os.path.splitext(k)[0]
        content = strip_frontmatter(get_text(user, k))
        if not content:
            continue
        add(f"### {name} (from C--Users-Amral)")
        add("")
        add(content)
        add("")

def chunk_debugging_protocol():
    add("## Debugging Protocol (Merged)")
    add("")

    for label, src in [
        ("C--Trading-bot/feedback", get_text(bot, "feedback/debugging-protocol.md")),
        ("crypto-agent-bot", get_text(crypto, "debugging-protocol.md")),
        ("crypto-agent-bot/user-debugging-style", get_text(crypto, "user-debugging-style.md")),
    ]:
        text = strip_frontmatter(src)
        if not text:
            continue
        add(f"### Source: {label}")
        add("")
        add(text)
        add("")

def chunk_project():
    add("## Project State & Milestones")
    add("")

    # Collect all project files from bot
    keys = sorted(k for k in bot if k.startswith("project/"))
    for key in keys:
        name = os.path.splitext(os.path.basename(key))[0]
        content = strip_frontmatter(bot[key])
        if not content:
            continue
        add(f"### {name}")
        add("")
        add(content)
        add("")

    # Add key crypto project files
    crypto_project_keys = [
        "project-overview.md",
        "active-debug-state.md",
        "planning-implementation-protocol.md",
        "graphify-knowledge-graph.md",
        "searxng-config.md",
    ]
    for k in crypto_project_keys:
        name = os.path.splitext(k)[0]
        content = strip_frontmatter(get_text(crypto, k, f"project/{k}"))
        if not content:
            continue
        # Avoid duplicating graphify and searxng already in bot
        if k in ["graphify-knowledge-graph.md", "searxng-config.md"]:
            bot_key = f"project/{k}"
            if bot_key in bot:
                # Append to existing
                continue
        add(f"### {name} (from crypto-agent-bot)")
        add("")
        add(content)
        add("")

    # User project files
    user_proj = get_text(user, "project-autoresearch-trading.md")
    if user_proj:
        add("### auto-research-trading (from C--Users-Amral)")
        add("")
        add(strip_frontmatter(user_proj))
        add("")

def chunk_reference():
    add("## References")
    add("")

    keys = sorted(k for k in bot if k.startswith("reference/"))
    for key in keys:
        name = os.path.splitext(os.path.basename(key))[0]
        content = strip_frontmatter(bot[key])
        if not content:
            continue
        add(f"### {name}")
        add("")
        add(content)
        add("")

def chunk_team():
    add("## Team Shared Knowledge")
    add("")

    keys = sorted(k for k in bot if k.startswith("team/"))
    for key in keys:
        name = os.path.splitext(os.path.basename(key))[0]
        content = strip_frontmatter(bot[key])
        if not content:
            continue
        add(f"### {name}")
        add("")
        add(content)
        add("")

# ===== BUILD =====
chunk_user_profile()
chunk_debugging_protocol()
chunk_feedback()
chunk_project()
chunk_reference()
chunk_team()

# ===== WRITE =====
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Written {len(lines)} lines ({sum(len(l) for l in lines)} chars) to {OUT}")
print(f"Wrote {len(bot)} bot keys, {len(crypto)} crypto keys, {len(user)} user keys")
