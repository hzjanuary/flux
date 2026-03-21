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
import time
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from config import SYSTEM_PROMPT_TEMPLATE, cfg
from memory import LambdaMemory, memory_manager
from tools import registry

logger = logging.getLogger(__name__)

# Maximum number of tool→LLM roundtrips per user message (prevents infinite loops)
MAX_TOOL_ITERATIONS = 5
LATENCY_WINDOW_SIZE = 30

# Heuristic: route likely "latest/current/news" questions through web-search first
REALTIME_QUERY_PATTERN = re.compile(
    r"\b(latest|current|news|update|today|breaking|moi nhat|mới nhất|tinh hinh|tình hình|hien tai|hiện tại|hom nay|hôm nay)\b",
    re.IGNORECASE,
)

# Heuristic for simple questions that can be answered directly without tools.
DIRECT_ANSWER_BLOCK_PATTERN = re.compile(
    r"\b(search|tìm|tra cứu|gia xang|giá xăng|hôm nay|today|latest|current|news|cpu|ram|disk|file|run|command|lệnh|save|lưu|note|ghi chú|shell|terminal|status latency)\b",
    re.IGNORECASE,
)


@dataclass
class TurnMetrics:
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    total_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    path: str = "react"
    model_used: str = ""


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
            timeout=30.0,
            max_retries=0,
        )
        # Tool manifest is static at runtime; cache once to avoid rebuilding each turn.
        self._tools_manifest = registry.get_tools_manifest()
        self._latency_by_chat = defaultdict(lambda: deque(maxlen=LATENCY_WINDOW_SIZE))
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
        t0 = time.perf_counter()
        metrics = TurnMetrics()

        # Step 1: Add user's message to short-term memory
        memory.add_message("user", user_message)

        # Step 2: route by intent for lower latency and lower token usage.
        if self._should_fast_path_web(user_message):
            metrics.path = "fast_web"
            final_response = await self._fast_path_web_answer(memory, user_message, metrics)
        elif self._should_direct_answer(user_message):
            metrics.path = "direct"
            final_response = await self._direct_answer(memory, user_message, metrics)
            # Guard: never leak tool-call JSON to user.
            if self._parse_tool_call(final_response):
                metrics.path = "react_fallback"
                final_response = await self._react_loop(memory, user_message, metrics)
        else:
            # Default path: full ReAct loop
            metrics.path = "react"
            final_response = await self._react_loop(memory, user_message, metrics)

        # Step 3: Store the final AI response in memory
        memory.add_message("assistant", final_response)

        # Step 4: fire-and-forget summarization to reduce user-visible latency.
        summarize_task = asyncio.create_task(memory.maybe_summarize(self.client))
        summarize_task.add_done_callback(self._on_background_task_done)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        metrics.total_ms = elapsed_ms
        self._record_latency(chat_id, metrics)
        logger.info("[Brain] think() completed for chat %d in %.1f ms", chat_id, elapsed_ms)

        return final_response

    @staticmethod
    def _on_background_task_done(task: asyncio.Task):
        """Log background-task failures without impacting user response flow."""
        try:
            task.result()
        except Exception:
            logger.exception("[Brain] Background task failed.")

    @staticmethod
    def _should_fast_path_web(user_message: str) -> bool:
        """Detect likely realtime/news intent so we can skip planning call."""
        if len(user_message) < 8:
            return False
        return bool(REALTIME_QUERY_PATTERN.search(user_message))

    @staticmethod
    def _should_direct_answer(user_message: str) -> bool:
        """Pick direct-answer path for short/simple prompts unlikely to need tools."""
        text = user_message.strip()
        if len(text) > 180:
            return False
        if DIRECT_ANSWER_BLOCK_PATTERN.search(text):
            return False
        # Keep it simple: short messages with small token budget
        return len(text.split()) <= 14

    async def _direct_answer(self, memory: LambdaMemory, user_message: str, metrics: TurnMetrics) -> str:
        """Answer simple questions directly with a compact prompt (no tool manifest)."""
        history = memory.get_history()[-2:]
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {cfg.AGENT_NAME}. Reply naturally in the user's language. "
                    "Keep answers concise and practical. "
                    "Do not output any JSON, and do not call tools."
                ),
            }
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = await self._call_llm(messages, metrics)
        if response:
            return response

        return (
            "⚠️ Tôi chưa thể trả lời nhanh lúc này. Tôi sẽ thử lại theo chế độ đầy đủ.\n"
            "Quick-answer mode was unavailable, retrying with full reasoning."
        )

    async def _fast_path_web_answer(self, memory: LambdaMemory, user_message: str, metrics: TurnMetrics) -> str:
        """
        Execute web search first, then ask LLM to synthesize answer.
        This reduces one model roundtrip for common realtime queries.
        """
        logger.info("[Brain] Fast-path web search enabled for message: %s", user_message[:80])
        tool_t0 = time.perf_counter()
        tool_result = await registry.execute(
            "search_web",
            {"query": user_message, "max_results": 4},
        )
        metrics.tool_ms += (time.perf_counter() - tool_t0) * 1000
        metrics.tool_calls += 1

        ephemeral_tool_messages = [
            {
                "role": "assistant",
                "content": (
                    '{"action":"TOOL_CALL","tool_name":"search_web","tool_args":'
                    f'{{"query":"{user_message}","max_results":4}},'
                    '"reasoning":"Realtime query fast-path."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    "[TOOL RESULT for `search_web`]\n"
                    f"{tool_result}\n\n"
                    "Hãy tổng hợp ngắn gọn, nêu nguồn chính và cảnh báo nếu thông tin chưa chắc chắn.\n"
                    "Summarize concisely, cite key sources, and mention uncertainty when needed."
                ),
            },
        ]

        messages = self._build_messages(memory, ephemeral_tool_messages)
        raw_response = await self._call_llm(messages, metrics)

        # Some models may still emit TOOL_CALL JSON even after receiving tool output.
        # Resolve up to 2 extra tool-calls to avoid leaking raw JSON to end users.
        for _ in range(2):
            if not raw_response:
                break
            tool_call = self._parse_tool_call(raw_response)
            if not tool_call:
                return raw_response

            tool_name = tool_call.get("tool_name", "")
            tool_args = tool_call.get("tool_args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            logger.info(
                "[Brain] Fast-path follow-up tool call: %s(%s)",
                tool_name,
                tool_args,
            )

            ephemeral_tool_messages.append({
                "role": "assistant",
                "content": raw_response,
            })
            tool_t0 = time.perf_counter()
            tool_result = await registry.execute(tool_name, tool_args)
            metrics.tool_ms += (time.perf_counter() - tool_t0) * 1000
            metrics.tool_calls += 1
            ephemeral_tool_messages.append({
                "role": "user",
                "content": (
                    f"[TOOL RESULT for `{tool_name}`]\n{tool_result}\n\n"
                    "Trả lời trực tiếp cho user, KHÔNG xuất JSON tool call.\n"
                    "Answer the user directly and DO NOT output any tool-call JSON."
                ),
            })

            messages = self._build_messages(memory, ephemeral_tool_messages)
            raw_response = await self._call_llm(messages, metrics)

        if raw_response and not self._parse_tool_call(raw_response):
            return raw_response

        return (
            "⚠️ Tôi đã lấy được dữ liệu web nhưng chưa tổng hợp được phản hồi từ AI. "
            "Vui lòng thử lại sau ít phút.\n"
            "I retrieved web data but could not synthesize a final AI response. "
            "Please try again shortly."
        )

    # ── ReAct Loop ─────────────────────────────────────────────────────────────

    async def _react_loop(self, memory: LambdaMemory, original_message: str, metrics: TurnMetrics) -> str:
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
            raw_response = await self._call_llm(messages, metrics)

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
                tool_t0 = time.perf_counter()
                tool_result = await registry.execute(tool_name, tool_args)
                metrics.tool_ms += (time.perf_counter() - tool_t0) * 1000
                metrics.tool_calls += 1
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

    async def _call_llm(self, messages: list, metrics: TurnMetrics) -> Optional[str]:
        """
        Make a single async call to the OpenRouter LLM API.

        Args:
            messages: The full list of message dicts for the API.

        Returns:
            The raw string content of the LLM's response, or None on error.
        """
        try:
            model_candidates = [cfg.DEFAULT_MODEL, *cfg.FALLBACK_MODELS]
            response = None
            used_model = ""

            for model in model_candidates:
                t0 = time.perf_counter()
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=cfg.MAX_TOKENS,
                        temperature=0.7,
                        # OpenRouter-specific headers (for rankings/attribution)
                        extra_headers={
                            "HTTP-Referer": "https://github.com/fluxclaw",
                            "X-Title": cfg.AGENT_NAME,
                        },
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    metrics.llm_ms += elapsed_ms
                    metrics.llm_calls += 1
                    metrics.model_used = model
                    logger.info("[Brain] LLM call completed in %.1f ms | model=%s", elapsed_ms, model)
                    used_model = model
                    break
                except Exception as model_err:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    metrics.llm_ms += elapsed_ms
                    metrics.llm_calls += 1
                    err_str = str(model_err)
                    if self._should_try_fallback(err_str):
                        logger.warning(
                            "[Brain] Model %s failed with fallback-eligible error: %s",
                            model,
                            err_str,
                        )
                        continue
                    raise model_err

            if response is None:
                logger.error("[Brain] All model candidates failed. Candidates: %s", model_candidates)
                return (
                    "⚠️ Tất cả model hiện đều không phản hồi (429/404). Vui lòng thử lại sau.\n"
                    "All configured models are unavailable right now (429/404). Please retry shortly."
                )

            if not getattr(response, "choices", None):
                logger.warning("[Brain] LLM response has no choices.")
                return (
                    "⚠️ AI trả về dữ liệu không hợp lệ (không có choices). Vui lòng thử lại.\n"
                    "The AI returned an invalid response (no choices). Please retry."
                )

            first_choice = response.choices[0]
            content = (getattr(first_choice.message, "content", None) or "").strip()
            # Guard: some providers return None content on edge cases
            if not content:
                logger.warning("[Brain] LLM returned empty content.")
                return (
                    "⚠️ AI trả về phản hồi rỗng. Vui lòng thử lại.\n"
                    "The AI returned an empty response. Please try again."
                )
            if used_model and used_model != cfg.DEFAULT_MODEL:
                logger.info("[Brain] Response generated by fallback model: %s", used_model)
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

    @staticmethod
    def _should_try_fallback(err_str: str) -> bool:
        lowered = err_str.lower()
        return (
            "429" in err_str
            or "404" in err_str
            or "rate limit" in lowered
            or "temporarily rate-limited" in lowered
            or "not found" in lowered
            or "no such model" in lowered
        )

    def _record_latency(self, chat_id: int, metrics: TurnMetrics):
        self._latency_by_chat[chat_id].append(metrics)

    def get_latency_report(self, chat_id: int) -> str:
        data = list(self._latency_by_chat.get(chat_id, []))
        if not data:
            return "📉 Chưa có dữ liệu latency cho chat này."

        total_turns = len(data)
        avg_total = sum(m.total_ms for m in data) / total_turns
        avg_llm = sum(m.llm_ms for m in data) / total_turns
        avg_tool = sum(m.tool_ms for m in data) / total_turns
        avg_llm_calls = sum(m.llm_calls for m in data) / total_turns
        avg_tool_calls = sum(m.tool_calls for m in data) / total_turns
        last = data[-1]

        return (
            f"⚡ Latency (last {total_turns} turns)\n"
            f"├── Avg Total: {avg_total:.1f} ms\n"
            f"├── Avg LLM: {avg_llm:.1f} ms ({avg_llm_calls:.2f} calls/turn)\n"
            f"├── Avg Tool: {avg_tool:.1f} ms ({avg_tool_calls:.2f} calls/turn)\n"
            f"└── Last Turn: total={last.total_ms:.1f} ms | llm={last.llm_ms:.1f} ms | "
            f"tool={last.tool_ms:.1f} ms | path={last.path} | model={last.model_used or cfg.DEFAULT_MODEL}"
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
            available_tools=self._tools_manifest,
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
