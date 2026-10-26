from unittest.mock import patch

import pytest

from agno.exceptions import ModelAuthenticationError
from agno.models.perplexity import PerplexityResponses


def test_perplexity_responses_default_config():
    model = PerplexityResponses()

    assert model.id == "perplexity/sonar"
    assert model.name == "PerplexityResponses"
    assert model.provider == "Perplexity"
    assert model.base_url == "https://api.perplexity.ai/v1"
    assert model.store is False


def test_perplexity_responses_requires_api_key():
    model = PerplexityResponses()

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ModelAuthenticationError, match="PERPLEXITY_API_KEY not set"):
            model._get_client_params()


def test_perplexity_responses_api_key_precedence():
    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "env-key"}):
        assert PerplexityResponses()._get_client_params()["api_key"] == "env-key"
        assert PerplexityResponses(api_key="explicit-key")._get_client_params()["api_key"] == "explicit-key"


def test_perplexity_responses_integration_header_keeps_custom_headers():
    model = PerplexityResponses(api_key="test-key", default_headers={"X-Custom": "1"})

    headers = model._get_client_params()["default_headers"]

    assert headers == {"X-Pplx-Integration": "agno", "X-Custom": "1"}


def test_perplexity_responses_omits_reasoning_unless_requested():
    assert "reasoning" not in PerplexityResponses(api_key="test-key").get_request_params()

    params = PerplexityResponses(id="openai/gpt-5.5", api_key="test-key", reasoning_effort="low").get_request_params()

    assert params["reasoning"] == {"effort": "low"}


def test_perplexity_responses_sets_max_output_tokens_for_anthropic_only():
    assert (
        PerplexityResponses(id="anthropic/claude-sonnet-5", api_key="k").get_request_params()["max_output_tokens"]
        == 8192
    )
    assert (
        PerplexityResponses(id="anthropic/claude-sonnet-5", api_key="k", max_output_tokens=512).get_request_params()[
            "max_output_tokens"
        ]
        == 512
    )
    assert "max_output_tokens" not in PerplexityResponses(id="openai/gpt-5.5", api_key="k").get_request_params()
