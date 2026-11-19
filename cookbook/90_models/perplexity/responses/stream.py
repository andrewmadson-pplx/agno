"""Streaming example using the Perplexity Agent API.

Requirements:
- Set PERPLEXITY_API_KEY environment variable
"""

from agno.agent import Agent
from agno.models.perplexity import PerplexityResponses

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

agent = Agent(
    model=PerplexityResponses(id="anthropic/claude-sonnet-5"),
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.print_response("Write a short poem about the moon", stream=True)
