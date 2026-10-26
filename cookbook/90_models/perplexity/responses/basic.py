"""Basic example using the Perplexity Agent API.

The Agent API is OpenAI Responses compatible and serves models from several labs
with one Perplexity key.

Requirements:
- Set PERPLEXITY_API_KEY environment variable
"""

import asyncio

from agno.agent import Agent
from agno.models.perplexity import PerplexityResponses

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

agent = Agent(
    model=PerplexityResponses(id="openai/gpt-5.5"),
    markdown=True,
)

# The string syntax works too: Agent(model="perplexity-responses:openai/gpt-5.5", markdown=True)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Sync ---
    agent.print_response("Share a 2 sentence horror story")

    # --- Async ---
    asyncio.run(agent.aprint_response("Share a 2 sentence horror story"))
