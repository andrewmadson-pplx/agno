"""
Multiple Workspace Providers — Tools Mode
=========================================

When context providers use mode=ContextMode.tools, they expose raw tools
directly instead of wrapping them in sub-agents. Without namespacing,
tools like `read_file` from different providers would collide.

Tool namespacing solves this: each provider prefixes tools with its id.

- WorkspaceContextProvider(id="code") exposes ``code_read_file``, ``code_grep_content``
- WorkspaceContextProvider(id="docs") exposes ``docs_read_file``, ``docs_grep_content``

The agent uses prefixed names to target the right source. This is useful
when you want direct tool access without sub-agent overhead.

Compare with: 08_multi_provider.py for default mode (sub-agents)
See also: 26_multi_calendar_tools_mode.py for calendar example

Requires: OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agno.agent import Agent
from agno.context.mode import ContextMode
from agno.context.workspace import WorkspaceContextProvider
from agno.models.openai import OpenAIResponses

# ---------------------------------------------------------------------------
# Create Context Providers
# ---------------------------------------------------------------------------
# Each provider gets a unique id that becomes the tool prefix.
# mode=ContextMode.tools exposes tools directly to the calling agent.

code = WorkspaceContextProvider(
    id="code",
    name="Source Code",
    root=Path(__file__).resolve().parents[2] / "libs" / "agno",
    mode=ContextMode.tools,
)

docs = WorkspaceContextProvider(
    id="docs",
    name="Documentation",
    root=Path(__file__).resolve().parents[2] / "cookbook",
    mode=ContextMode.tools,
)

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
# Combine tools from both providers. No collision because of prefixing.

agent = Agent(
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[*code.get_tools(), *docs.get_tools()],
    instructions=f"{code.instructions()}\n\n{docs.instructions()}",
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("\n" + "=" * 60)
    print("Multi-Provider Tools Mode Demo")
    print("=" * 60)

    print("\nAvailable tools (notice the prefixes):")
    for toolkit in code.get_tools():
        for name in toolkit.functions:
            print(f"  - {name}")
    for toolkit in docs.get_tools():
        for name in toolkit.functions:
            print(f"  - {name}")

    prompt = (
        "Use code_grep_content to find where 'prefix_tools_with_name' is defined, "
        "then use docs_list_files to see what's in cookbook/12_context."
    )
    print(f"\n> {prompt}\n")
    await agent.aprint_response(prompt, stream=True)


if __name__ == "__main__":
    asyncio.run(main())
