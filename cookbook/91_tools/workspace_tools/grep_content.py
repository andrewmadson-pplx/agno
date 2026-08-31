"""
Workspace — grep_content
========================

Regex search with line numbers across text files. Returns JSON with
file:line citations for direct navigation.

This example demonstrates a rename pre-flight sweep: finding all
occurrences of a method name before refactoring, using word-boundary
regex to avoid false positives.

Requires: OPENAI_API_KEY
"""

from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.workspace import Workspace

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
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
if __name__ == "__main__":
    prompt = (
        "We're planning to rename the sync method `search_content` "
        "(the async variant `asearch_content` keeps its name).\n\n"
        "1. Search with the word-boundary pattern `\\bsearch_content\\b` and 1 context line\n"
        "2. Classify each occurrence as: definition, call site, docstring/comment, or test\n\n"
        "Produce a rename checklist with file:line citations."
    )
    print(f"> {prompt}\n")
    agent.print_response(prompt)
