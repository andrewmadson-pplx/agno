# TEST_LOG

All examples run against the live Perplexity Agent API (`https://api.perplexity.ai/v1`)
with `PERPLEXITY_API_KEY` set, on 2026-10-27. Python 3.12, `agno[dev]` in a fresh `.venv`.

### basic.py

**Status:** PASS

**Description:** Sync and async `print_response` runs of `openai/gpt-5.5` through
`PerplexityResponses`.

**Result:** Both runs returned a two sentence story with no API errors.

---

### stream.py

**Status:** PASS

**Description:** Streaming `print_response` run of `anthropic/claude-sonnet-5`. Exercises the
default `max_output_tokens` the class adds for `anthropic/*` models.

**Result:** Poem streamed in chunks, no API errors.

---

### tool_use.py

**Status:** PASS

**Description:** Streaming agent run with `PerplexitySearch` tools on `openai/gpt-5.5`.

**Result:** The model called `search(...)` once, then answered with a summary of current news
and source URLs. No API errors.

---

### structured_output.py

**Status:** PASS

**Description:** `output_schema=MovieScript` on `openai/gpt-5.5`.

**Result:** Returned a valid `MovieScript` object with all fields populated.

---
