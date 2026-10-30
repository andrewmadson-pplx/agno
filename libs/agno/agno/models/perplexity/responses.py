from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from agno.exceptions import ModelAuthenticationError
from agno.models.message import Message
from agno.models.openai.open_responses import OpenResponses
from agno.run.agent import RunOutput


@dataclass
class PerplexityResponses(OpenResponses):
    """
    A class for interacting with models through the Perplexity Agent API.

    The Agent API is OpenAI Responses compatible and serves models from several labs
    under one key, with Perplexity's built-in `web_search` and `fetch_url` tools.
    Requests are stateless.

    For more information, see: https://docs.perplexity.ai/docs/agent-api/quickstart

    Attributes:
        id (str): The model id. Defaults to "perplexity/sonar".
        name (str): The model name. Defaults to "PerplexityResponses".
        provider (str): The provider name. Defaults to "Perplexity".
        api_key (Optional[str]): The API key. Uses PERPLEXITY_API_KEY env var if not set.
        base_url (str): The base URL. Defaults to "https://api.perplexity.ai/v1".

    Example:
        ```python
        from agno.agent import Agent
        from agno.models.perplexity import PerplexityResponses

        agent = Agent(
            model=PerplexityResponses(id="openai/gpt-5.5"),
            markdown=True,
        )
        agent.print_response("Write a haiku about coding")
        ```
    """

    id: str = "perplexity/sonar"
    name: str = "PerplexityResponses"
    provider: str = "Perplexity"

    api_key: Optional[str] = None
    base_url: str = "https://api.perplexity.ai/v1"

    store: Optional[bool] = False

    # The Agent API requires max_output_tokens for Anthropic models; same default as the Claude class
    anthropic_max_output_tokens: int = 8192

    def _set_reasoning_request_param(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Only send a `reasoning` block when the caller explicitly opts in, since not
        every Agent API model accepts one.
        """
        if self.reasoning is None and self.reasoning_effort is None and self.reasoning_summary is None:
            return base_params

        reasoning: Dict[str, Any] = dict(self.reasoning or {})
        if self.reasoning_effort is not None:
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            reasoning["summary"] = self.reasoning_summary

        base_params["reasoning"] = reasoning
        return base_params

    def _get_client_params(self) -> Dict[str, Any]:
        """
        Returns client parameters for API requests, checking for PERPLEXITY_API_KEY.

        Returns:
            Dict[str, Any]: A dictionary of client parameters for API requests.

        Raises:
            ModelAuthenticationError: If PERPLEXITY_API_KEY is not set.
        """
        if not self.api_key:
            self.api_key = getenv("PERPLEXITY_API_KEY")
            if not self.api_key:
                raise ModelAuthenticationError(
                    message="PERPLEXITY_API_KEY not set. Please set the PERPLEXITY_API_KEY environment variable.",
                    model_name=self.name,
                )

        base_params: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": {"X-Pplx-Integration": "agno", **(self.default_headers or {})},
            "default_query": self.default_query,
        }

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params

    def get_request_params(
        self,
        messages: Optional[List[Message]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> Dict[str, Any]:
        request_params = super().get_request_params(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
        )
        if self.id.startswith("anthropic/") and request_params.get("max_output_tokens") is None:
            request_params["max_output_tokens"] = self.anthropic_max_output_tokens
        return request_params

    def _using_reasoning_model(self) -> bool:
        """OpenAI reasoning models served by the Agent API use OpenAI's reasoning format."""
        return self.id.startswith(("openai/o3", "openai/o4", "openai/gpt-5", "openai/gpt-6"))
