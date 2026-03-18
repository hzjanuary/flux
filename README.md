<div align="center">

# FluxClaw

**A personal AI Agent on Telegram — Powered by OpenRouter**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Telethon](https://img.shields.io/badge/Telethon-1.36%2B-blue?logo=telegram)](https://github.com/LonamiWebs/Telethon)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Combining the power of **PicoClaw** (modular), **OpenClaw** (tool calling) and **SkyClaw** (Lambda Memory)*

</div>

---

## Introduction

**FluxClaw** is a personal AI Agent running on Telegram, built on the **ReAct (Reasoning + Acting)** architecture. The agent runs as a **Telegram Bot** (via Bot Token from `@BotFather`) and is capable of:

- Conversing naturally in **Vietnamese and English**
- Automatically **calling tools** (web search, system info, note saving, math, etc.)
- Maintaining long-term context via the **Lambda Memory** system (fully async)
- Supporting **multiple users** with isolated memory per chat
- **Persisting state** — memory survives bot restarts

---

## Architecture

```
flux/
├── config.py        — API keys, environment variables, system prompt template
├── memory.py        — Lambda Memory (async STM + LLM-based summarization)
├── tools.py         — ToolRegistry + tool implementations
├── brain.py         — ReAct loop + OpenRouter integration
├── bot.py           — Telethon event loop (Telegram Bot interface)
├── .env             — Secret configuration (do NOT commit to git)
├── .env.example     — Configuration template
└── requirements.txt — Python dependencies
```

### Processing Flow (ReAct Loop)

```
Message from Telegram
        |
   [bot.py] receives event → is_authorized() check
        |
   [brain.py] brain.think()
        |
   Build prompt: System Prompt + Core Context (LTM) + Short-term history
        |
   Call LLM (OpenRouter) ----> TOOL_CALL? ----> [tools.py] Execute tool
                          ^____________________|  Feed result back to LLM
        |
   (final plain-text answer)
   [memory.py] Update STM → async summarization if full
        |
   Send reply to Telegram
```

---

## Lambda Memory System

Memory operates on **2 tiers**, summarization is **fully async** (non-blocking):

| Tier | Name | Description |
|------|------|-------------|
| Tier 1 | Short-Term Memory (STM) | Deque holding the last **N messages** (default: 10) |
| Tier 2 | Core Context (LTM) | LLM-generated summary of older history, injected into every System Prompt |

When STM is full, the **oldest half** is summarized by the LLM (via `AsyncOpenAI`) and merged into Core Context. The **newest half** stays in STM for immediate context.

All memory is saved to `fluxclaw_memory.json` and **persists across restarts**.

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `get_system_info()` | CPU, RAM, disk usage, and OS information |
| `get_current_time()` | Current date and time |
| `search_web(query, max_results)` | Real web search via DuckDuckGo (title + link + snippet) |
| `write_note(title, content)` | Save a text note to the `notes/` directory |
| `read_file(filepath)` | Read file contents (max 50KB) |
| `run_shell_command(command)` | Run whitelisted safe shell commands |
| `calculate(expression)` | Evaluate a math expression safely |

> Adding new tools is straightforward — see the guide below.

---

## Setup

### Requirements

- Python **3.10+**
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)
- Telegram App credentials from [my.telegram.org](https://my.telegram.org)
- An API Key from [OpenRouter](https://openrouter.ai/keys)

### Step 1 — Get Telegram App Credentials

1. Go to [https://my.telegram.org/auth](https://my.telegram.org/auth)
2. Log in and select **"API Development Tools"**
3. Create an application and save your `api_id` and `api_hash`

> These are required by Telethon regardless of login mode.

### Step 2 — Create a Bot via @BotFather

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the **Bot Token** (format: `123456789:AAxxxxxx...`)

### Step 3 — Configure `.env`

```bash
copy .env.example .env
```

Open `.env` and fill in your details:

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here

# Bot Token from @BotFather
TELEGRAM_BOT_TOKEN=123456789:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
DEFAULT_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Strongly recommended: set to your Telegram User ID
TELEGRAM_OWNER_ID=0
```

> **Security tip:** Set `TELEGRAM_OWNER_ID` to your Telegram User ID (get it from [@userinfobot](https://t.me/userinfobot)) to prevent others from using your bot and consuming your OpenRouter API quota.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the bot

```bash
py bot.py
```

No OTP or phone number required. The bot connects immediately using the Bot Token and starts listening for messages.

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and introduction |
| `/help` | List all available tools |
| `/status` | View memory state (STM count + Core Context preview) |
| `/reset` | Clear all memory for the current chat |

---

## Adding a New Tool

Open `tools.py` and append to the bottom of the file:

```python
@registry.register(
    description="Description of what this tool does — the LLM reads this to decide when to use it",
    params={
        "param1": {"type": "string", "description": "Description of the parameter"}
    }
)
async def my_tool_name(param1: str) -> str:
    # Your logic here
    return f"Result: {param1}"
```

Done. The tool is **automatically** included in the system prompt and available for the agent to call.

---

## Security

| Concern | Mitigation |
|---------|------------|
| Unauthorized access | Set `TELEGRAM_OWNER_ID` — bot logs a warning at startup if left as `0` |
| Bot Token exposure | Never commit `.env` to git (`.gitignore` already covers this) |
| Session file leakage | `*.session` files hold auth data — do not share or commit |
| Shell command abuse | `run_shell_command` uses an **allowlist** — only safe read-only commands |
| Prompt injection via files | `read_file` caps at **50KB** |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_API_ID` | Yes | — | Telegram App ID from my.telegram.org |
| `TELEGRAM_API_HASH` | Yes | — | Telegram App Hash from my.telegram.org |
| `TELEGRAM_BOT_TOKEN` | Yes | — | Bot Token from @BotFather |
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API Key |
| `TELEGRAM_PHONE` | No | `""` | Phone number (UserBot mode only, unused by default) |
| `TELEGRAM_OWNER_ID` | No | `0` (public) | Restrict bot to a single Telegram User ID |
| `DEFAULT_MODEL` | No | `google/gemini-2.0-flash-001` | Primary LLM model |
| `SUMMARIZER_MODEL` | No | `google/gemini-2.0-flash-001` | Memory summarization model |
| `MAX_TOKENS` | No | `2048` | Maximum response token limit |
| `SHORT_TERM_LIMIT` | No | `10` | Max messages in short-term memory before summarization |
| `AGENT_NAME` | No | `FluxClaw` | Agent display name |
| `SESSION_NAME` | No | `fluxclaw_session` | Telethon session filename |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [telethon](https://github.com/LonamiWebs/Telethon) | Telegram Client API (Bot Token mode) |
| [openai](https://github.com/openai/openai-python) | OpenRouter-compatible async LLM SDK |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Load environment variables from `.env` |
| [psutil](https://github.com/giampaolo/psutil) | System information (CPU, RAM, disk) |
| [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) | Async real web search via DuckDuckGo |

---

## Design Influences

| Source Project | Contribution |
|----------------|-------------|
| **PicoClaw** | Clean module structure, clear separation of concerns |
| **OpenClaw** | ToolRegistry pattern, ReAct loop, function calling logic |
| **SkyClaw (Temm1e)** | Lambda Memory: sliding window + async LLM summarization |

---

<div align="center">

Made with care · Powered by [OpenRouter](https://openrouter.ai) · Built with [Telethon](https://github.com/LonamiWebs/Telethon)

</div>
