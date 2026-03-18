"""
bot.py — Fusion-Jarvis Telegram Interface
==========================================
The main entry point. Creates the Telethon client, registers event handlers,
and manages the event loop.

Architecture:
    Telethon Client (Telegram)
         ↓ (NewMessage event)
    handle_message()
         ↓
    brain.think(chat_id, text) → [ReAct Loop in brain.py]
         ↓
    Send response back via Telegram

Commands:
    /start  — Welcome message
    /reset  — Clear all memory for this chat
    /status — Show memory stats
    /help   — List available tools

Usage:
    python bot.py
"""

import asyncio
import logging
import sys

from telethon import TelegramClient, events
from telethon.tl.types import User

from brain import brain
from config import cfg
from memory import memory_manager
from tools import registry

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fusion_jarvis.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Telethon Client ────────────────────────────────────────────────────────────
# The session file (cfg.SESSION_NAME + ".session") is created on first run
# and reused on subsequent starts to avoid re-authentication.
client = TelegramClient(
    cfg.SESSION_NAME,
    cfg.TELEGRAM_API_ID,
    cfg.TELEGRAM_API_HASH,
)

# ── Typing Indicator: Show "typing..." while the AI thinks ────────────────────
async def send_typing(chat_id: int):
    """Sends a 'typing' action to show the bot is processing."""
    try:
        async with client.action(chat_id, "typing"):
            await asyncio.sleep(99)  # Will be cancelled when response is ready
    except asyncio.CancelledError:
        pass


# ── Helper: Check if message is from owner (optional security) ────────────────
def is_authorized(event) -> bool:
    """
    Returns True if TELEGRAM_OWNER_ID is 0 (public bot) or
    if the message sender matches the owner ID.
    """
    if cfg.TELEGRAM_OWNER_ID == 0:
        return True  # No restriction — respond to everyone
    return event.sender_id == cfg.TELEGRAM_OWNER_ID


# ══════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@client.on(events.NewMessage(pattern="/start"))
async def handle_start(event):
    """Welcome message handler."""
    if not is_authorized(event):
        return

    sender = await event.get_sender()
    name = sender.first_name if isinstance(sender, User) else "bạn"

    welcome = (
        f"👋 Xin chào **{name}**! Tôi là **{cfg.AGENT_NAME}** — AI Assistant của bạn.\n\n"
        f"Hello **{name}**! I'm **{cfg.AGENT_NAME}** — your intelligent AI Assistant.\n\n"
        f"🧠 **Khả năng / Capabilities:**\n"
        f"• Trả lời câu hỏi bằng tiếng Việt & tiếng Anh\n"
        f"• Sử dụng {len(registry.list_tools())} công cụ tích hợp sẵn\n"
        f"• Ghi nhớ cuộc trò chuyện với Lambda Memory\n\n"
        f"💡 **Lệnh / Commands:** /help | /status | /reset\n\n"
        f"Hãy bắt đầu nhắn tin! Just start chatting! 🚀"
    )
    await event.reply(welcome)


@client.on(events.NewMessage(pattern="/help"))
async def handle_help(event):
    """Display available tools and usage information."""
    if not is_authorized(event):
        return

    tools_manifest = registry.get_tools_manifest()
    help_text = (
        f"🛠️ **{cfg.AGENT_NAME} — Danh sách công cụ / Available Tools**\n\n"
        f"{tools_manifest}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💬 Bạn không cần gọi tool trực tiếp — chỉ cần mô tả yêu cầu bằng ngôn ngữ tự nhiên!\n"
        f"You don't need to call tools directly — just describe your request naturally!\n\n"
        f"**Ví dụ / Examples:**\n"
        f"• \"Máy tính của tôi đang dùng bao nhiêu RAM?\"\n"
        f"• \"Search for Python async best practices\"\n"
        f"• \"Tính 2^32 là bao nhiêu?\"\n"
        f"• \"Lưu note: Cần mua sữa\""
    )
    await event.reply(help_text)


@client.on(events.NewMessage(pattern="/status"))
async def handle_status(event):
    """Show memory statistics for this chat."""
    if not is_authorized(event):
        return

    chat_id = event.chat_id
    memory = memory_manager.get(chat_id)
    stm_count = len(memory.short_term)
    has_core = bool(memory.core_context)
    core_preview = (
        memory.core_context[:150] + "..."
        if len(memory.core_context) > 150
        else memory.core_context or "(empty)"
    )

    status = (
        f"📊 **Memory Status — Chat {chat_id}**\n\n"
        f"├── 🔵 Short-Term Memory: {stm_count}/{cfg.SHORT_TERM_LIMIT} messages\n"
        f"├── 🟣 Core Context (Long-Term): {'✅ Active' if has_core else '❌ Empty'}\n"
        f"└── 📝 Core Preview:\n`{core_preview}`\n\n"
        f"🤖 Model: `{cfg.DEFAULT_MODEL}`\n"
        f"🔧 Tools: {len(registry.list_tools())} registered"
    )
    await event.reply(status)


@client.on(events.NewMessage(pattern="/reset"))
async def handle_reset(event):
    """Clear all memory for this chat."""
    if not is_authorized(event):
        return

    chat_id = event.chat_id
    memory_manager.clear(chat_id)
    await event.reply(
        "🗑️ **Đã xóa toàn bộ bộ nhớ!**\n"
        "All memory has been cleared! Starting fresh. 🆕"
    )


@client.on(events.NewMessage)
async def handle_message(event):
    """
    Main message handler — processes all non-command messages.

    Flow:
        1. Ignore bots, commands, and unauthorized users
        2. Show typing indicator
        3. Pass message to brain.think() → ReAct loop
        4. Send response
        5. Cancel typing indicator
    """
    # Skip if it's a command (handled by specific handlers above)
    if event.text and event.text.startswith("/"):
        return

    # Skip if sender is a bot or if not authorized
    sender = await event.get_sender()
    if isinstance(sender, User) and sender.bot:
        return
    if not is_authorized(event):
        logger.info("[Bot] Ignored message from unauthorized user: %s", event.sender_id)
        return

    # Skip empty messages or media-only messages
    if not event.text or not event.text.strip():
        return

    chat_id = event.chat_id
    user_text = event.text.strip()

    logger.info(
        "[Bot] Message from chat %d: %s",
        chat_id, user_text[:100]
    )

    # Start 'typing' indicator in background
    typing_task = asyncio.create_task(send_typing(chat_id))

    try:
        # 🧠 Core processing — the magic happens here
        response = await brain.think(chat_id, user_text)

        # Telegram message limit is 4096 chars; split if necessary
        if len(response) > 4090:
            chunks = [response[i:i+4090] for i in range(0, len(response), 4090)]
            for i, chunk in enumerate(chunks):
                suffix = f"\n\n_(Part {i+1}/{len(chunks)})_" if len(chunks) > 1 else ""
                await event.reply(chunk + suffix)
        else:
            await event.reply(response)

    except Exception as e:
        logger.exception("[Bot] Unhandled error processing message from chat %d.", chat_id)
        await event.reply(
            f"❌ Đã xảy ra lỗi không mong muốn: `{type(e).__name__}`\n"
            f"An unexpected error occurred. Please try again."
        )
    finally:
        # Always cancel the typing indicator when done
        typing_task.cancel()


# ── Main Entry Point ───────────────────────────────────────────────────────────

async def main():
    """
    Start the Telethon client and begin listening for messages.
    On first run, Telethon will prompt for your phone number and OTP.
    """
    logger.info("=" * 60)
    logger.info("  %s — Starting Up", cfg.AGENT_NAME)
    logger.info("  Model: %s", cfg.DEFAULT_MODEL)
    logger.info("  Memory limit: %d messages per chat", cfg.SHORT_TERM_LIMIT)
    logger.info("  Tools: %s", ", ".join(registry.list_tools()))
    logger.info("=" * 60)

    await client.start(phone=cfg.TELEGRAM_PHONE)

    me = await client.get_me()
    logger.info(
        "[Bot] Logged in as: %s (@%s) [ID: %s]",
        me.first_name, me.username, me.id
    )

    if cfg.TELEGRAM_OWNER_ID:
        logger.info("[Bot] Restricted to owner ID: %d", cfg.TELEGRAM_OWNER_ID)
    else:
        logger.info("[Bot] Running in PUBLIC mode (responding to all users).")

    print(f"\n✅ {cfg.AGENT_NAME} is online and listening for messages...")
    print("   Press Ctrl+C to stop.\n")

    # Keep the client running until manually stopped
    await client.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[Bot] Shutdown requested by user (Ctrl+C). Goodbye!")
