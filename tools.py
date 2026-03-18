"""
tools.py — FluxClaw Tool Registry
========================================
Defines all tools (Python functions) that the AI agent can call.

Architecture (ToolRegistry Pattern):
    1. Each tool is a regular Python async function decorated with @registry.register()
    2. The decorator captures metadata (name, description, parameter schema)
    3. The Brain module queries registry.get_tools_manifest() to tell the LLM
       what tools are available (injected into the system prompt)
    4. When the LLM decides to use a tool, brain.py calls registry.execute(name, args)

Adding a new tool:
    1. Define an async function with type-annotated parameters
    2. Decorate it with @registry.register(description="...", params={...})
    3. That's it — it's automatically available to the agent!
"""

import asyncio
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

import psutil  # For system info (pip install psutil)
from duckduckgo_search import DDGS  # For web search (pip install duckduckgo-search)

logger = logging.getLogger(__name__)


# ── Tool Metadata ──────────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """Metadata for a single tool, including the function to execute."""
    name: str
    description: str
    # JSON Schema for the parameters (sent to LLM to know what args to provide)
    parameters: Dict[str, Any]
    # The actual async Python function
    fn: Callable[..., Coroutine]


# ── Tool Registry ──────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all agent tools.

    Responsibilities:
        - Store tool definitions and their implementations
        - Provide a manifest (list of tool descriptions) for the LLM system prompt
        - Execute tools by name with given arguments
        - Handle execution errors gracefully
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        description: str,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """
        Decorator factory: registers a function as an agent tool.

        Args:
            description: Human/LLM-readable description of what this tool does.
            params: JSON Schema-style parameter definitions.
            name: Optional override for the tool name (defaults to function name).

        Example:
            @registry.register(
                description="Gets the current time",
                params={"timezone": {"type": "string", "description": "e.g. 'UTC', 'Asia/Ho_Chi_Minh'"}}
            )
            async def get_time(timezone: str = "UTC") -> str:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=description,
                parameters=params or {},
                fn=fn,
            )
            logger.debug("[ToolRegistry] Registered tool: '%s'", tool_name)
            return fn
        return decorator

    def get_tools_manifest(self) -> str:
        """
        Generate a formatted string listing all available tools.
        This is injected into the LLM system prompt so the AI knows what it can do.

        Returns:
            A formatted string like:
            "- get_system_info(): Returns CPU, RAM usage..."
            "- write_note(title, content): Saves a note to disk..."
        """
        if not self._tools:
            return "No tools available."
        lines = []
        for tool in self._tools.values():
            params_str = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in tool.parameters.items()
            )
            lines.append(f"- **{tool.name}**({params_str}): {tool.description}")
        return "\n".join(lines)

    async def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Look up and execute a tool by name.

        Args:
            tool_name: The name of the tool to run.
            tool_args: Arguments to pass to the tool function.

        Returns:
            A string result from the tool (always stringified for LLM consumption).
        """
        tool = self._tools.get(tool_name)
        if not tool:
            available = ", ".join(self._tools.keys())
            return f"❌ Tool '{tool_name}' not found. Available: [{available}]"

        logger.info("[ToolRegistry] Executing: %s(%s)", tool_name, tool_args)
        try:
            result = await tool.fn(**tool_args)
            return str(result)
        except TypeError as e:
            return f"❌ Invalid arguments for '{tool_name}': {e}"
        except Exception as e:
            logger.exception("[ToolRegistry] Tool '%s' raised an error.", tool_name)
            return f"❌ Tool '{tool_name}' failed with error: {type(e).__name__}: {e}"

    def list_tools(self) -> List[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())


# ── Global Registry Instance ───────────────────────────────────────────────────
registry = ToolRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# Add new tools below this line. They are auto-registered on module import.
# ══════════════════════════════════════════════════════════════════════════════


@registry.register(
    description=(
        "Lấy thông tin hệ thống hiện tại: CPU, RAM, disk, OS, thời gian. "
        "Use when user asks about system performance or machine info."
    ),
)
async def get_system_info() -> str:
    """Returns a detailed snapshot of the current machine's hardware and OS status."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    boot_time = datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info = (
        f"🖥️ **System Info** (as of {now})\n"
        f"├── OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"├── Python: {platform.python_version()}\n"
        f"├── CPU Usage: {cpu_percent}%\n"
        f"├── RAM: {ram.percent}% used "
        f"({ram.used / 1e9:.1f}GB / {ram.total / 1e9:.1f}GB)\n"
        f"├── Disk: {disk.percent}% used "
        f"({disk.used / 1e9:.1f}GB / {disk.total / 1e9:.1f}GB)\n"
        f"└── System Up Since: {boot_time}"
    )
    return info


@registry.register(
    description=(
        "Lấy ngày giờ hiện tại. Use when user asks about current time or date."
    ),
)
async def get_current_time() -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return (
        f"🕐 **Current Time**\n"
        f"├── Date: {now.strftime('%A, %B %d, %Y')}\n"
        f"├── Time: {now.strftime('%H:%M:%S')}\n"
        f"└── ISO: {now.isoformat()}"
    )


@registry.register(
    description=(
        "Tìm kiếm web bằng DuckDuckGo (không cần API key). "
        "Use when user asks you to search for something online or look up current information."
    ),
    params={
        "query": {
            "type": "string",
            "description": "The search query string (e.g., 'Python 3.12 new features')",
        },
        "max_results": {
            "type": "integer",
            "description": "Number of results to return (default: 5, max: 10)",
        },
    },
)
async def search_web(query: str, max_results: int = 5) -> str:
    """
    Searches the web via DuckDuckGo using the duckduckgo_search library.
    Runs the sync DDGS client in a thread pool executor to avoid blocking
    the async event loop.
    Returns real search results: title, URL, and a content snippet per result.
    """
    max_results = min(max_results, 10)

    def _blocking_search() -> list:
        """Synchronous search — safe to run inside an executor thread."""
        with DDGS() as ddgs:
            return ddgs.text(query, max_results=max_results)

    try:
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, _blocking_search)

        if not raw_results:
            return (
                f"🔍 No results found for '{query}'. "
                f"Try: https://duckduckgo.com/?q={query.replace(' ', '+')}"
            )

        lines = [f"🔍 **Search results for: '{query}'**\n"]
        for i, result in enumerate(raw_results, start=1):
            title = result.get("title", "No title")
            href  = result.get("href", "")
            body  = result.get("body", "").strip()
            # Truncate long snippets to keep the prompt concise
            snippet = body[:300] + "…" if len(body) > 300 else body
            lines.append(
                f"{i}. **{title}**\n"
                f"   🔗 {href}\n"
                f"   {snippet}"
            )

        return "\n\n".join(lines)

    except Exception as e:
        logger.error("[search_web] Failed: %s", e)
        return f"❌ Search failed: {type(e).__name__}: {e}"


@registry.register(
    description=(
        "Lưu một ghi chú văn bản vào file. "
        "Use when user asks you to save, write, or remember something as a note."
    ),
    params={
        "title": {
            "type": "string",
            "description": "The note title (used as filename, keep it short and descriptive)",
        },
        "content": {
            "type": "string",
            "description": "The full text content of the note",
        },
    },
)
async def write_note(title: str, content: str) -> str:
    """Saves a text note to the 'notes/' directory with a timestamp."""
    notes_dir = Path("notes")
    notes_dir.mkdir(exist_ok=True)

    # Sanitize title for use as filename
    safe_title = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title
    ).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = notes_dir / f"{safe_title}_{timestamp}.txt"

    file_content = (
        f"=== {title} ===\n"
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'─' * 40}\n"
        f"{content}\n"
    )

    try:
        filename.write_text(file_content, encoding="utf-8")
        return f"✅ Note saved: `{filename}`"
    except Exception as e:
        return f"❌ Failed to save note: {e}"


@registry.register(
    description=(
        "Đọc nội dung của một file text. "
        "Use when user asks you to read or show the contents of a specific file."
    ),
    params={
        "filepath": {
            "type": "string",
            "description": "Relative or absolute path to the file to read",
        },
    },
)
async def read_file(filepath: str) -> str:
    """Reads and returns the content of a text file."""
    path = Path(filepath)
    if not path.exists():
        return f"❌ File not found: `{filepath}`"
    if not path.is_file():
        return f"❌ Path is not a file: `{filepath}`"

    # Safety: limit file size to 50KB to prevent prompt flooding
    if path.stat().st_size > 50_000:
        return f"❌ File too large to read (>{50_000 / 1000:.0f}KB): `{filepath}`"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return f"📄 **Contents of `{filepath}`:**\n\n```\n{content}\n```"
    except Exception as e:
        return f"❌ Could not read file: {e}"


@registry.register(
    description=(
        "Chạy một lệnh shell/terminal an toàn (whitelist). "
        "Use when user asks to run a system command. ONLY safe commands are allowed."
    ),
    params={
        "command": {
            "type": "string",
            "description": "The shell command to execute (e.g., 'ls -la', 'echo hello')",
        },
    },
)
async def run_shell_command(command: str) -> str:
    """
    Executes a whitelisted shell command and returns its output.
    Safety: Only commands starting with allowed prefixes will be executed.
    """
    # Security whitelist — only allow safe read-only-ish commands
    ALLOWED_PREFIXES = [
        "ls", "dir", "echo", "cat", "pwd", "whoami",
        "python --version", "pip list", "pip show",
        "git log", "git status", "git diff",
    ]

    is_safe = any(command.strip().startswith(p) for p in ALLOWED_PREFIXES)
    if not is_safe:
        return (
            f"🚫 Command blocked for safety: `{command}`\n"
            f"Allowed prefixes: {', '.join(ALLOWED_PREFIXES)}"
        )

    try:
        result = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=15.0)
        output = stdout.decode("utf-8", errors="replace").strip()
        error = stderr.decode("utf-8", errors="replace").strip()

        response = f"💻 **Command:** `{command}`\n\n"
        if output:
            response += f"**Output:**\n```\n{output}\n```"
        if error:
            response += f"\n**Stderr:**\n```\n{error}\n```"
        return response or "✅ Command ran with no output."
    except asyncio.TimeoutError:
        return f"❌ Command timed out (>15s): `{command}`"
    except Exception as e:
        return f"❌ Command failed: {e}"


@registry.register(
    description=(
        "Tính toán biểu thức toán học. "
        "Use when user asks you to calculate or evaluate a math expression."
    ),
    params={
        "expression": {
            "type": "string",
            "description": "A Python-safe math expression (e.g., '2 ** 10 + 5 * 3')",
        },
    },
)
async def calculate(expression: str) -> str:
    """Safely evaluates a math expression using Python's eval with restricted scope."""
    import math

    # Whitelist of safe names for eval (no builtins, no imports)
    safe_globals = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "pow": pow, "divmod": divmod, "len": len,
        "math": math,
        "pi": math.pi, "e": math.e, "inf": math.inf,
    }

    try:
        result = eval(expression, safe_globals)  # noqa: S307 (controlled eval)
        return f"🧮 `{expression}` = **{result}**"
    except ZeroDivisionError:
        return "❌ Division by zero."
    except SyntaxError:
        return f"❌ Invalid expression syntax: `{expression}`"
    except Exception as e:
        return f"❌ Calculation error: {e}"
