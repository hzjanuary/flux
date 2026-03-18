"""
config.py — FluxClaw Configuration Center
================================================
Manages all API keys and environment variables using python-dotenv.
All sensitive credentials are loaded from a .env file and NEVER hardcoded.

Usage:
    from config import cfg
    print(cfg.OPENROUTER_API_KEY)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file from the project root directory
load_dotenv()


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object. All settings are loaded once at startup
    from environment variables (populated by .env file).
    """

    # ── Telegram Credentials ──────────────────────────────────────────────────
    # Obtain from https://my.telegram.org/auth → API Development Tools
    TELEGRAM_API_ID: int = field(
        default_factory=lambda: int(os.environ["TELEGRAM_API_ID"])
    )
    TELEGRAM_API_HASH: str = field(
        default_factory=lambda: os.environ["TELEGRAM_API_HASH"]
    )
    # Phone number — only used in UserBot mode (leave empty when using Bot Token)
    TELEGRAM_PHONE: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_PHONE", "")
    )
    # Bot Token from @BotFather — used instead of phone number for Bot API mode
    TELEGRAM_BOT_TOKEN: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    # Optional: restrict the bot to only respond to messages from this user ID
    TELEGRAM_OWNER_ID: int = field(
        default_factory=lambda: int(os.environ.get("TELEGRAM_OWNER_ID", "0"))
    )

    # ── OpenRouter / LLM Settings ─────────────────────────────────────────────
    # Obtain from https://openrouter.ai/keys
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.environ["OPENROUTER_API_KEY"]
    )
    OPENROUTER_BASE_URL: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )
    # Default model to use. Can be overridden per-call.
    # Examples: "anthropic/claude-3.5-sonnet", "google/gemini-pro", "openai/gpt-4o"
    DEFAULT_MODEL: str = field(
        default_factory=lambda: os.environ.get(
            "DEFAULT_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
        )
    )
    # Model used exclusively for summarizing memory (can be a cheaper/faster model)
    SUMMARIZER_MODEL: str = field(
        default_factory=lambda: os.environ.get(
            "SUMMARIZER_MODEL", "google/gemini-2.0-flash-001"
        )
    )
    # Maximum tokens for the main LLM response
    MAX_TOKENS: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TOKENS", "2048"))
    )

    # ── Memory Settings ───────────────────────────────────────────────────────
    # Number of recent messages to keep in short-term memory before summarizing
    SHORT_TERM_LIMIT: int = field(
        default_factory=lambda: int(os.environ.get("SHORT_TERM_LIMIT", "10"))
    )

    # ── Agent Identity ────────────────────────────────────────────────────────
    AGENT_NAME: str = field(
        default_factory=lambda: os.environ.get("AGENT_NAME", "FluxClaw")
    )
    AGENT_LANGUAGE: str = field(
        default_factory=lambda: os.environ.get("AGENT_LANGUAGE", "Vietnamese/English")
    )

    # ── Session File ─────────────────────────────────────────────────────────
    # Telethon saves session data here to avoid re-login on every restart
    SESSION_NAME: str = field(
        default_factory=lambda: os.environ.get("SESSION_NAME", "fluxclaw_session")
    )


# ── Singleton instance ────────────────────────────────────────────────────────
# Import this object in all other modules: `from config import cfg`
cfg = Config()


# ── System Prompt Template ────────────────────────────────────────────────────
# This is the "soul" of FluxClaw, injected into every LLM request.
# {core_context} will be replaced dynamically by the Lambda Memory system.
SYSTEM_PROMPT_TEMPLATE = """
Bạn là {agent_name} — một AI Assistant thông minh, mạnh mẽ và thân thiện.
You are {agent_name} — an intelligent, powerful, and friendly AI Assistant.

## Core Identity
- Bạn nói chuyện tự nhiên bằng tiếng Việt và tiếng Anh (ngôn ngữ mà user dùng).
- You naturally converse in Vietnamese and English (match the user's language).
- Phong cách: Thông minh, hữu ích, đôi khi hài hước, luôn tôn trọng.
- Style: Intelligent, helpful, occasionally witty, always respectful.

## Tool Usage Protocol (ReAct Loop)
Khi bạn cần thực hiện một hành động (tìm kiếm, lấy thông tin hệ thống, v.v.),
bạn PHẢI trả lời theo định dạng JSON đặc biệt này và CHỈ có JSON này thôi:

When you need to perform an action (search, get system info, etc.),
you MUST respond with this special JSON format and NOTHING ELSE:

```json
{{
  "action": "TOOL_CALL",
  "tool_name": "<tool_name>",
  "tool_args": {{
    "<arg_name>": "<arg_value>"
  }},
  "reasoning": "<brief explanation of why you are calling this tool>"
}}
```

Available tools: {available_tools}

Nếu bạn KHÔNG cần tool, hãy trả lời tự nhiên như bình thường.
If you do NOT need a tool, respond naturally as normal.

## Long-Term Memory (Core Context)
{core_context}
""".strip()
