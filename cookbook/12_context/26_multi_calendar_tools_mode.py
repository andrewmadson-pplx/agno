"""
Multiple Calendars — Tools Mode
===============================

Two Google Calendar providers with mode=ContextMode.tools: one for work,
one for personal. Each provider prefixes its tools with its id:

- ``work_list_events``, ``work_get_event``, ``work_search_events``
- ``personal_list_events``, ``personal_get_event``, ``personal_search_events``

The agent can query both calendars simultaneously without tool name collisions.
Use case: "Am I free Thursday?" checks both work and personal calendars.

Compare with: 19_calendar.py for single calendar (default mode)
See also: 25_multi_provider_tools_mode.py for workspace example

Setup (use different credentials for each calendar):
    Work calendar (service account)::

        export GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/work-sa.json

    Personal calendar (OAuth)::

        export GOOGLE_CLIENT_ID=...
        export GOOGLE_CLIENT_SECRET=...
        export GOOGLE_PROJECT_ID=...

    Or use two service accounts with different env var names.

Requires: OPENAI_API_KEY + Google Calendar credentials
"""

from __future__ import annotations

import asyncio

from agno.agent import Agent
from agno.context.calendar import GoogleCalendarContextProvider
from agno.context.mode import ContextMode
from agno.models.openai import OpenAIResponses

# ---------------------------------------------------------------------------
# Create Context Providers
# ---------------------------------------------------------------------------
# Each calendar gets a unique id that becomes the tool prefix.

work = GoogleCalendarContextProvider(
    id="work",
    name="Work Calendar",
    mode=ContextMode.tools,
)

personal = GoogleCalendarContextProvider(
    id="personal",
    name="Personal Calendar",
    mode=ContextMode.tools,
    token_path="personal_calendar_token.json",
)

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
# Combine tools from both calendars. No collision because of prefixing.

agent = Agent(
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[*work.get_tools(), *personal.get_tools()],
    instructions=(
        "You have access to two calendars:\n\n"
        f"{work.instructions()}\n\n"
        f"{personal.instructions()}\n\n"
        "When checking availability, query BOTH calendars. "
        "Use work_* tools for work calendar, personal_* tools for personal."
    ),
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("\n" + "=" * 60)
    print("Multi-Calendar Tools Mode Demo")
    print("=" * 60)

    print("\nWork calendar tools:")
    for toolkit in work.get_tools():
        for name in toolkit.functions:
            print(f"  - {name}")

    print("\nPersonal calendar tools:")
    for toolkit in personal.get_tools():
        for name in toolkit.functions:
            print(f"  - {name}")

    prompt = "Am I free Thursday afternoon? Check both calendars."
    print(f"\n> {prompt}\n")
    await agent.aprint_response(prompt, stream=True)


if __name__ == "__main__":
    asyncio.run(main())
