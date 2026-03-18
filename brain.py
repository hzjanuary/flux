"""
brain.py — FluxClaw Reasoning Engine
==========================================
Integrates OpenRouter API and implements the full ReAct (Reasoning + Acting) loop.

ReAct Loop Flow:
    1. Build context: system prompt (with core memory) + short-term history + new user msg
    2. Call LLM → parse response
    3. If response is a TOOL_CALL JSON → execute tool → feed result back to LLM (Step 2)
    4. If response is plain text → that's the final answer
    5. Update memory with all turns
    6. Trigger async memory summarization check

This module is intentionally decoupled from Telegram — it only deals with
text in / text out, making it easy to test or swap the communication layer.
"""

import json
import logging
import re
from typing import Optional

from openai import AsyncOpenAI

from config import SYSTEM_PROMPT_TEMPLATE, cfg
from memory import LambdaMemory, memory_manager
from tools import registry

logger = logging.getLogger(__name__)

# Maximum number of tool→LLM roundtrips per user message (prevents infinite loops)
MAX_TOOL_ITERATIONS = 5


class Brain:
    """
    The reasoning engine of FluxClaw.

    Responsibilities:
        - Maintain the AsyncOpenAI client (OpenRouter-compatible)
        - Build the full prompt on each turn (system + memory + history + new message)
        - Run the ReAct loop (call LLM → detect tool call → execute → repeat)
        - Delegate memory reads/writes to LambdaMemory
        - Return the final natural-language response to the caller
    """

    def __init__(self):
        # AsyncOpenAI client pointed at OpenRouter's API
        self.client = AsyncOpenAI(
            api_key=cfg.OPENROUTER_API_KEY,
            base_url=cfg.OPENROUTER_BASE_URL,
        )
        logger.info(
            "[Brain] Initialized. Model: %s | Summarizer: %s",
            cfg.DEFAULT_MODEL, cfg.SUMMARIZER_MODEL
        )

    # ── Public Entry Point ─────────────────────────────────────────────────────

    async def think(self, chat_id: int, user_message: str) -> str:
        """
        Process a user message through the full ReAct loop.

        Args:
            chat_id: Telegram chat ID (used to route to correct memory).
            user_message: The raw text sent by the user.

        Returns:
            The final response string to send back to the user.
        """
        memory = memory_manager.get(chat_id)

        # Step 1: Add user's message to short-term memory
        memory.add_message("user", user_message)

        # Step 2: Run the ReAct loop
        final_response = await self._react_loop(memory, user_message)

        # Step 3: Store the final AI response in memory
        memory.add_message("assistant", final_response)

        # Step 4: Async summarization check — reuse the shared AsyncOpenAI client,
        # no extra client creation, no blocking of the event loop.
        await memory.maybe_summarize(self.client)

        return final_response

    # ── ReAct Loop ─────────────────────────────────────────────────────────────

    async def _react_loop(self, memory: LambdaMemory, original_message: str) -> str:
        """
        The core ReAct (Reasoning + Acting) loop.

        Iterations:
            - iter 0: LLM sees user message → may return TOOL_CALL
            - iter 1: LLM sees tool result → may return another TOOL_CALL or final answer
            - ... (up to MAX_TOOL_ITERATIONS)
            - On final iter or plain-text response: return the answer

        Args:
            memory: The LambdaMemory for this chat.
            original_message: The original user text (used for fallback context).

        Returns:
            The final response string.
        """
        # Accumulate tool interaction messages for this turn
        # These are added to the LLM context but NOT persisted in memory
        # (only the final Q&A pair is stored in STM)
        ephemeral_tool_messages = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            logger.debug("[Brain] ReAct iteration %d/%d", iteration + 1, MAX_TOOL_ITERATIONS)

            # Build the full message list for this LLM call
            messages = self._build_messages(memory, ephemeral_tool_messages)

            # Call the LLM — always returns a string (error messages included)
            raw_response = await self._call_llm(messages)

            # Guard: if somehow empty, return immediately
            if not raw_response:
                return (
                    "❌ Xin lỗi, tôi gặp lỗi kết nối với AI. Vui lòng thử lại.\n"
                    "Sorry, I encountered a connection error. Please try again."
                )

            # Try to parse as a TOOL_CALL
            tool_call = self._parse_tool_call(raw_response)

            if tool_call:
                # ── ACTING phase ──────────────────────────────────────────────
                tool_name = tool_call.get("tool_name", "")
                tool_args = tool_call.get("tool_args", {})
                reasoning = tool_call.get("reasoning", "")

                logger.info(
                    "[Brain] Tool call detected: %s(%s) | Reason: %s",
                    tool_name, tool_args, reasoning
                )

                # Add the assistant's tool-call decision as context for next turn
                ephemeral_tool_messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                # Execute the tool
                tool_result = await registry.execute(tool_name, tool_args)
                logger.debug("[Brain] Tool result for '%s': %s", tool_name, tool_result[:200])

                # Add the tool result as a "tool" role message
                ephemeral_tool_messages.append({
                    "role": "user",  # Feed result back as user context
                    "content": (
                        f"[TOOL RESULT for `{tool_name}`]\n{tool_result}\n\n"
                        "Bây giờ hãy trả lời user dựa trên kết quả trên.\n"
                        "Now answer the user based on the result above."
                    ),
                })
                # Continue loop → LLM will see the tool result and generate final answer

            else:
                # ── REASONING phase done — final answer found ──────────────
                logger.debug("[Brain] Final answer found at iteration %d.", iteration + 1)
                return raw_response

        # If we exhausted all iterations without a final answer
        logger.warning("[Brain] Exceeded MAX_TOOL_ITERATIONS (%d).", MAX_TOOL_ITERATIONS)
        return (
            "⚠️ Tôi đã cố gắng nhiều bước nhưng không thể hoàn thành yêu cầu. "
            "Vui lòng thử lại hoặc đặt câu hỏi khác.\n\n"
            "I tried multiple steps but couldn't complete the request. "
            "Please try again or rephrase."
        )

    # ── LLM Call ───────────────────────────────────────────────────────────────

    async def _call_llm(self, messages: list) -> Optional[str]:
        """
        Make a single async call to the OpenRouter LLM API.

        Args:
            messages: The full list of message dicts for the API.

        Returns:
            The raw string content of the LLM's response, or None on error.
        """
        try:
            response = await self.client.chat.completions.create(
                model=cfg.DEFAULT_MODEL,
                messages=messages,
                max_tokens=cfg.MAX_TOKENS,
                temperature=0.7,
                # OpenRouter-specific headers (for rankings/attribution)
                extra_headers={
                    "HTTP-Referer": "https://github.com/fluxclaw",
                    "X-Title": cfg.AGENT_NAME,
                },
            )
            content = response.choices[0].message.content
            # Guard: some providers return None content on edge cases
            if not content or not content.strip():
                logger.warning("[Brain] LLM returned empty content.")
                return (
                    "⚠️ AI trả về phản hồi rỗng. Vui lòng thử lại.\n"
                    "The AI returned an empty response. Please try again."
                )
            logger.debug("[Brain] LLM response (first 200 chars): %s", content[:200])
            return content

        except Exception as e:
            err_str = str(e)
            logger.error("[Brain] LLM API call failed: %s", err_str)

            # ── 429: Rate limit or quota exceeded ─────────────────────────
            if "429" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
                return (
                    "⚠️ **Hệ thống AI đang quá tải hoặc đã hết quota.**\n"
                    "Vui lòng thử lại sau vài giây hoặc đổi model trong `.env`.\n\n"
                    "⚠️ **The AI service is rate-limited or quota exceeded.**\n"
                    "Please wait a moment and try again, or switch to a different model."
                )
            # ── 401: Invalid API key ───────────────────────────────────────
            if "401" in err_str or "unauthorized" in err_str.lower() or "user not found" in err_str.lower():
                return (
                    "❌ **API Key không hợp lệ hoặc chưa được cấu hình.**\n"
                    "Vui lòng kiểm tra `OPENROUTER_API_KEY` trong file `.env`.\n\n"
                    "❌ **Invalid or missing API Key.**\n"
                    "Please check `OPENROUTER_API_KEY` in your `.env` file."
                )
            # ── Generic network / unknown error ───────────────────────────
            return (
                "❌ Xin lỗi, tôi gặp lỗi kết nối với AI. Vui lòng thử lại.\n"
                "Sorry, I encountered a connection error. Please try again."
            )

    # ── Message Builder ────────────────────────────────────────────────────────

    def _build_messages(
        self,
        memory: LambdaMemory,
        ephemeral_tool_messages: list,
    ) -> list:
        """
        Construct the full message list for an LLM API call.

        Structure:
            [0] System message (agent identity + core context + tool manifest)
            [1..N] Short-term memory history (past turns)
            [N+1..M] Ephemeral tool interaction messages (current turn tool calls)

        Args:
            memory: Current chat's LambdaMemory.
            ephemeral_tool_messages: Tool call/result pairs for the current turn.

        Returns:
            A list of message dicts ready for the OpenAI API.
        """
        # Build the system prompt with injected memory and tool list
        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=cfg.AGENT_NAME,
            available_tools=registry.get_tools_manifest(),
            core_context=memory.get_core_context_block(),
        )

        messages = [
            {"role": "system", "content": system_content}
        ]

        # Inject short-term memory history
        messages.extend(memory.get_history())

        # Inject any ephemeral tool messages for this turn
        messages.extend(ephemeral_tool_messages)

        return messages

    # ── Tool Call Parser ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_tool_call(response_text: str) -> Optional[dict]:
        """
        Attempt to parse the LLM's response as a TOOL_CALL JSON.

        The LLM is instructed to respond with a specific JSON format when it
        wants to call a tool. This method extracts that JSON.

        Handles:
            - Pure JSON responses
            - JSON wrapped in ```json ... ``` code fences
            - Mixed text with embedded JSON (scans for the JSON block)

        Args:
            response_text: The raw string from the LLM.

        Returns:
            A dict with "tool_name", "tool_args", etc. if valid,
            or None if this is not a tool call.
        """
        if not response_text:
            return None

        # Strategy 1: Try the whole response as JSON
        stripped = response_text.strip()
        try:
            data = json.loads(stripped)
            if data.get("action") == "TOOL_CALL" and "tool_name" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from ```json ... ``` fences
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            response_text,
            re.DOTALL,
        )
        if fence_match:
            try:
                data = json.loads(fence_match.group(1))
                if data.get("action") == "TOOL_CALL" and "tool_name" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find any raw JSON object in the response
        json_match = re.search(r"\{[^{}]*\"action\"\s*:\s*\"TOOL_CALL\"[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if data.get("action") == "TOOL_CALL" and "tool_name" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Not a tool call — it's a natural language response
        return None


# ── Singleton Instance ─────────────────────────────────────────────────────────
brain = Brain()
