"""
memory.py — Lambda Memory System
=================================
Implements the sliding-window + summarization memory architecture.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Short-Term Memory (Ring Buffer, max N messages)           │
    │  [msg1, msg2, ..., msg10]                                  │
    │       ↓ (when full)                                        │
    │  Summarizer LLM call → Core Context (Long-Term Memory)     │
    │  {core_context} injected into every System Prompt          │
    └─────────────────────────────────────────────────────────────┘

Persistence:
    Memory is saved to a JSON file so context survives bot restarts.
"""

import json
import logging
import os
import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

from openai import AsyncOpenAI

from config import cfg

logger = logging.getLogger(__name__)

# Path to persist memory to disk
MEMORY_FILE = "fluxclaw_memory.json"


@dataclass
class Message:
    """Represents a single conversation turn."""
    role: str   # "user", "assistant", or "tool"
    content: str
    # Optional: which tool call this message relates to (for "tool" role)
    tool_name: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to OpenAI-compatible message dict."""
        d = {"role": self.role, "content": self.content}
        # Tool result messages use "tool" role with a name field
        if self.tool_name:
            d["name"] = self.tool_name
        return d


class LambdaMemory:
    """
    Lambda Memory: A two-tier memory system.

    Tier 1 — Short-Term Memory (STM):
        A deque (ring buffer) holding the last N conversation turns.
        When it overflows, the oldest messages are summarized.

    Tier 2 — Core Context (Long-Term Memory):
        A text summary of all previous conversations, injected into
        every new system prompt to preserve long-term coherence.

    Per-chat isolation:
        Each Telegram chat/user gets their own independent memory instance,
        keyed by chat_id to prevent context bleeding between users.
    """

    def __init__(self, chat_id: int):
        """
        Initialize memory for a specific chat.

        Args:
            chat_id: The Telegram chat ID (used as the unique memory key).
        """
        self.chat_id = chat_id
        self.short_term: Deque[Message] = deque(maxlen=cfg.SHORT_TERM_LIMIT)
        self.core_context: str = ""  # Long-term summary
        self._summary_lock = asyncio.Lock()
        self._load_from_disk()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, tool_name: Optional[str] = None):
        """
        Append a new message to short-term memory.
        The deque's maxlen automatically evicts the oldest message when full.

        Args:
            role: "user", "assistant", or "tool"
            content: The message text / tool result.
            tool_name: Name of the tool (only for role="tool").
        """
        self.short_term.append(
            Message(role=role, content=content, tool_name=tool_name)
        )
        logger.debug(
            "[Memory] Added [%s] message. STM size: %d/%d",
            role, len(self.short_term), cfg.SHORT_TERM_LIMIT
        )

    def get_history(self) -> List[Dict]:
        """
        Return the short-term memory as a list of OpenAI-compatible dicts.
        This is fed directly into the LLM messages array.
        """
        return [msg.to_dict() for msg in self.short_term]

    async def maybe_summarize(self, openai_client: AsyncOpenAI):
        """
        Check if short-term memory is full. If so, summarize the oldest
        half of messages and merge the summary into core_context.

        This is the key "Lambda" trigger — called after every turn.
        Fully async: does NOT block the event loop while awaiting the LLM.

        Args:
            openai_client: The shared AsyncOpenAI client from Brain.
        """
        if len(self.short_term) < cfg.SHORT_TERM_LIMIT:
            return  # Not full yet, nothing to do

        async with self._summary_lock:
            # Re-check after lock acquisition because another task may have summarized already.
            if len(self.short_term) < cfg.SHORT_TERM_LIMIT:
                return

            logger.info(
                "[Memory] STM full (%d msgs) for chat %d. Triggering summarization...",
                cfg.SHORT_TERM_LIMIT, self.chat_id
            )

            # Take the OLDEST half to summarize, keep the NEWEST half in STM
            all_messages = list(self.short_term)
            half = len(all_messages) // 2
            to_summarize = all_messages[:half]
            to_keep = all_messages[half:]

            # Build the conversation text for summarization
            conversation_text = "\n".join(
                f"[{m.role.upper()}]: {m.content}" for m in to_summarize
            )

            # Compose the summarizer prompt
            summarize_prompt = (
                f"Đây là đoạn hội thoại cần tóm tắt:\n\n{conversation_text}\n\n"
                f"Context hiện tại:\n{self.core_context}\n\n"
                "Hãy tóm tắt ngắn gọn những điểm quan trọng nhất: các facts được đề cập, "
                "quyết định của user, và bất kỳ ngữ cảnh nào cần nhớ. "
                "Viết bằng ngôn ngữ mà user đang dùng (Việt/Anh).\n\n"
                "---\n"
                "Summarize the key facts, user decisions, and important context from this "
                "conversation. Be concise. Merge with or update the existing context above."
            )

            try:
                # ✅ Fully async — does not block the event loop
                response = await openai_client.chat.completions.create(
                    model=cfg.SUMMARIZER_MODEL,
                    messages=[{"role": "user", "content": summarize_prompt}],
                    max_tokens=512,
                )
                new_summary = response.choices[0].message.content.strip()
                self.core_context = new_summary
                logger.info("[Memory] Summarization complete. Core context updated.")
            except Exception as e:
                logger.error("[Memory] Summarization failed: %s", e)
                # On failure, retain the old core_context — graceful degradation

            # Replace STM with only the newer half (the older half was summarized)
            self.short_term = deque(to_keep, maxlen=cfg.SHORT_TERM_LIMIT)
            self._save_to_disk()

    def get_core_context_block(self) -> str:
        """
        Returns a formatted string for injection into the system prompt.
        Returns an empty placeholder if no long-term context exists yet.
        """
        if not self.core_context:
            return "Chưa có lịch sử cuộc trò chuyện nào. (No prior conversation history.)"
        return f"### Tóm tắt lịch sử (Memory Summary):\n{self.core_context}"

    def clear(self):
        """Reset all memory for this chat (useful for /reset command)."""
        self.short_term.clear()
        self.core_context = ""
        self._save_to_disk()
        logger.info("[Memory] Memory cleared for chat %d.", self.chat_id)

    # ── Persistence (Disk I/O) ─────────────────────────────────────────────────

    def _save_to_disk(self):
        """Serialize and save all memory to a JSON file."""
        try:
            all_data = self._load_raw_data()
            all_data[str(self.chat_id)] = {
                "short_term": [asdict(m) for m in self.short_term],
                "core_context": self.core_context,
            }
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("[Memory] Failed to save to disk: %s", e)

    def _load_from_disk(self):
        """Load memory from disk on startup (restores state after restart)."""
        try:
            all_data = self._load_raw_data()
            chat_data = all_data.get(str(self.chat_id), {})
            if chat_data:
                raw_messages = chat_data.get("short_term", [])
                self.short_term = deque(
                    [
                        Message(
                            role=m["role"],
                            content=m["content"],
                            tool_name=m.get("tool_name"),
                        )
                        for m in raw_messages
                    ],
                    maxlen=cfg.SHORT_TERM_LIMIT,
                )
                self.core_context = chat_data.get("core_context", "")
                logger.info(
                    "[Memory] Loaded %d messages + core context for chat %d.",
                    len(self.short_term), self.chat_id
                )
        except Exception as e:
            logger.warning("[Memory] Could not load from disk: %s", e)

    @staticmethod
    def _load_raw_data() -> Dict:
        """Load the raw JSON file if it exists."""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


# ── Memory Manager (Global Registry) ──────────────────────────────────────────
class MemoryManager:
    """
    A global registry that creates and caches one LambdaMemory instance
    per chat_id. This ensures proper per-user memory isolation.

    Usage:
        memory = memory_manager.get(chat_id)
        memory.add_message("user", "Hello!")
    """

    def __init__(self):
        self._registry: Dict[int, LambdaMemory] = {}

    def get(self, chat_id: int) -> LambdaMemory:
        """Get or create memory for a given chat."""
        if chat_id not in self._registry:
            self._registry[chat_id] = LambdaMemory(chat_id)
        return self._registry[chat_id]

    def clear(self, chat_id: int):
        """Clear and remove memory for a given chat."""
        if chat_id in self._registry:
            self._registry[chat_id].clear()
            del self._registry[chat_id]


# Singleton instance used across the application
memory_manager = MemoryManager()
