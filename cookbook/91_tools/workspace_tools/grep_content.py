"""
Workspace — grep_content
========================

grep_content is for precise code search: regex patterns, every matching
line with its line number, and optional context lines. Returns JSON with
file:line citations for direct navigation.

This example demonstrates a rename pre-flight sweep: finding all
occurrences of a method name before refactoring, using word-boundary
regex to avoid false positives.

Compare with: search_content (substring match, per-file results)
See also: 13_workspace.py for WorkspaceContextProvider

Requires: OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.workspace import Workspace

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
# Workspace with read-only tools: read, list, search, grep.
# confirm=[] disables confirmation prompts for read operations.

project_root = Path(__file__).resolve().parents[3]
libs_dir = project_root / "libs" / "agno" / "agno"

agent = Agent(
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[
        Workspace(
            libs_dir,
            allowed=["read", "list", "search", "grep"],
            confirm=[],
        )
    ],
    instructions=(
        "You are a refactoring assistant helping prepare a method rename.\n\n"
        "Workflow:\n"
        "1. Use grep_content to find all occurrences with line numbers\n"
        "2. Use context_lines if you need surrounding code to classify matches\n"
        "3. Use read_file with a line range when you need more context\n\n"
        "Use word-boundary regex (\\b) to avoid false positives. For example,\n"
        "`\\bsearch_content\\b` matches the sync method but not `asearch_content`.\n\n"
        "Cite every finding as file:line so the developer can navigate directly."
    ),
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("\n" + "=" * 60)
    print("grep_content Demo — Rename Pre-flight Sweep")
    print("=" * 60)

    prompt = (
        "We're planning to rename the sync method `search_content` "
        "(the async variant `asearch_content` keeps its name).\n\n"
        "1. Search with the word-boundary pattern `\\bsearch_content\\b` and 1 context line\n"
        "2. Classify each occurrence as: definition, call site, docstring/comment, or test\n\n"
        "Produce a rename checklist with file:line citations."
    )
    print(f"\n> {prompt}\n")
    await agent.aprint_response(prompt, stream=True)


if __name__ == "__main__":
    asyncio.run(main())
