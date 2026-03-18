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

**Flux** is a personal AI Agent running on Telegram, built on the **ReAct (Reasoning + Acting)** architecture. The agent is capable of:

- Conversing naturally in **Vietnamese and English**
- Automatically **calling tools** (web search, system info, note saving, etc.)
- Maintaining long-term context via the **Lambda Memory** system
- Supporting **multiple users** with isolated memory per chat
- **Persisting state** — memory survives bot restarts

---

## Architecture

```
flux/
├── config.py        — API keys, environment variables, system prompt template
├── memory.py        — Lambda Memory (STM + LLM-based summarization)
├── tools.py         — ToolRegistry + tool implementations
├── brain.py         — ReAct loop + OpenRouter integration
├── bot.py           — Telethon event loop (Telegram interface)
├── .env             — Secret configuration (do NOT commit to git)
├── .env.example     — Configuration template
└── requirements.txt — Python dependencies
```

### Processing Flow (ReAct Loop)

```
Message from Telegram
        |
   [bot.py] receives event
        |
   [brain.py] brain.think()
        |
   Build prompt: System Prompt + Core Context + Short-term history
        |
   Call LLM (OpenRouter) ----> TOOL_CALL? ----> [tools.py] Execute tool
                          ^____________________|  Feed result back to LLM
        |
   (final answer)
   [memory.py] Update memory
        |
   Send reply to Telegram
```

---

## Lambda Memory System

Memory operates on **2 tiers**:

| Tier | Name | Description |
|------|------|-------------|
| Tier 1 | Short-Term Memory (STM) | Deque holding the last **10 messages** |
| Tier 2 | Core Context (LTM) | LLM-generated summary of older history, injected into every System Prompt |

When STM is full, the LLM automatically summarizes the older half and stores it in Core Context.
All memory is saved to `fluxclaw_memory.json` and **persists across restarts**.

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `get_system_info()` | CPU, RAM, disk, and OS information |
| `get_current_time()` | Current date and time |
| `search_web(query)` | DuckDuckGo search (no API key required) |
| `write_note(title, content)` | Save a text note to the `notes/` directory |
| `read_file(filepath)` | Read file contents (max 50KB) |
| `run_shell_command(command)` | Run whitelisted safe shell commands |
| `calculate(expression)` | Evaluate a math expression safely |

> Adding new tools is straightforward — see the guide below.

---

## Setup

### Requirements

- Python **3.10+**
- A personal Telegram account
- An API Key from [OpenRouter](https://openrouter.ai/keys)

### Step 1 — Get Telegram API Credentials

1. Go to [https://my.telegram.org/auth](https://my.telegram.org/auth)
2. Log in and select **"API Development Tools"**
3. Create an application and save your `api_id` and `api_hash`

### Step 2 — Configure `.env`

```bash
copy .env.example .env
```

Open `.env` and fill in your details:

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here

# Phone number MUST be in international format
# Example: 0123456789 → drop the leading 0, add +84 → +84123456789
TELEGRAM_PHONE=+84123456789

OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
DEFAULT_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

> **Phone number format:**
> `0123456789` → remove leading `0` → prepend `+84` → `+84123456789`

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the bot

```bash
py bot.py
```

On the first run, Telethon will send an OTP to your Telegram app. Enter the code in the terminal. The session is then saved and reused on subsequent starts.

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

Done. The tool is automatically included in the system prompt and available for the agent to use.

---

## Security

- Set `TELEGRAM_OWNER_ID` to your Telegram User ID to restrict the bot to respond only to you.
  *(Get your ID by messaging [@userinfobot](https://t.me/userinfobot))*
- `*.session` files contain authentication tokens — **do not share or commit them**
- Shell commands are restricted to a **whitelist** — only safe, read-oriented commands are allowed
- File reading is capped at **50KB** to prevent prompt injection

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_API_ID` | Yes | — | Telegram API ID |
| `TELEGRAM_API_HASH` | Yes | — | Telegram API Hash |
| `TELEGRAM_PHONE` | Yes | — | Phone number in `+84...` format |
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API Key |
| `TELEGRAM_OWNER_ID` | No | `0` (public) | Restrict bot to a single user ID |
| `DEFAULT_MODEL` | No | `meta-llama/llama-3.3-70b-instruct:free` | Primary LLM model |
| `SUMMARIZER_MODEL` | No | `google/gemini-2.0-flash-001` | Memory summarization model |
| `MAX_TOKENS` | No | `2048` | Maximum response token limit |
| `SHORT_TERM_LIMIT` | No | `10` | Max messages in short-term memory |
| `AGENT_NAME` | No | `fluxclaw` | Agent display name |
| `SESSION_NAME` | No | `fluxclaw_session` | Telethon session filename |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| [telethon](https://github.com/LonamiWebs/Telethon) | Telegram Client API |
| [openai](https://github.com/openai/openai-python) | OpenRouter-compatible LLM SDK |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Load environment variables from `.env` |
| [psutil](https://github.com/giampaolo/psutil) | System information (CPU, RAM, disk) |
| [httpx](https://www.python-httpx.org/) | Async HTTP client (web search) |

---

## Design Influences

| Source Project | Contribution |
|----------------|-------------|
| **PicoClaw** | Clean module structure, clear separation of concerns |
| **OpenClaw** | ToolRegistry pattern, ReAct loop, function calling logic |
| **SkyClaw (Temm1e)** | Lambda Memory: sliding window + LLM summarization |

---

<div align="center">

Made with care · Powered by [OpenRouter](https://openrouter.ai) · Built with [Telethon](https://github.com/LonamiWebs/Telethon)

</div>
