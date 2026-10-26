"""Tool use example using the Perplexity Agent API.

The agent calls the Perplexity Search API through PerplexitySearch tools.

Requirements:
- Set PERPLEXITY_API_KEY environment variable
"""

from agno.agent import Agent
from agno.models.perplexity import PerplexityResponses
from agno.tools.perplexity import PerplexitySearch

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

agent = Agent(
    model=PerplexityResponses(id="openai/gpt-5.5"),
    tools=[PerplexitySearch()],
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.print_response("What is the latest news about AI?", stream=True)
