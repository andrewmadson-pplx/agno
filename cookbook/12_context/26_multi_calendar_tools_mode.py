"""
Multiple Calendars — Tools Mode
===============================

Two Google Calendar providers with mode=ContextMode.tools: one for work,
one for personal. Each provider prefixes its tools with its id:

- ``work_list_events``, ``work_get_event``, ``work_search_events``
- ``personal_list_events``, ``personal_get_event``, ``personal_search_events``

The agent can query both calendars simultaneously without tool name collisions.

Setup:
    Work calendar (service account)::
        export GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/work-sa.json

    Personal calendar (OAuth)::
        export GOOGLE_CLIENT_ID=...
        export GOOGLE_CLIENT_SECRET=...
        export GOOGLE_PROJECT_ID=...

Requires: OPENAI_API_KEY + Google Calendar credentials
"""

from agno.agent import Agent
from agno.context.calendar import GoogleCalendarContextProvider
from agno.context.mode import ContextMode
from agno.models.openai import OpenAIResponses

# ---------------------------------------------------------------------------
# Create Context Providers
# ---------------------------------------------------------------------------
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
if __name__ == "__main__":
    print("Work calendar tools:")
    for toolkit in work.get_tools():
        for name in toolkit.functions:
            print(f"  - {name}")

    print("\nPersonal calendar tools:")
    for toolkit in personal.get_tools():
        for name in toolkit.functions:
            print(f"  - {name}")

    print("\nNote: This example requires Google Calendar credentials.")
    print("Set up service account or OAuth credentials, then run:")
    print("  agent.print_response('Am I free Thursday afternoon?')")
