import base64
from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Tuple

from agno.knowledge.embedder.base import (
    Embedder,
    aembed_texts_individually,
    first_embedding,
    pad_batch_embeddings,
    raise_embedding_error,
)
from agno.utils.log import log_warning, logger

try:
    import requests
except ImportError:
    raise ImportError("`requests` not installed, use pip install requests")

try:
    import aiohttp
except ImportError:
    raise ImportError("`aiohttp` not installed, use pip install aiohttp")


def decode_int8_embedding(encoded: str) -> List[float]:
    """Decode a `base64_int8` Perplexity embedding into floats."""
    return [float(b - 256 if b > 127 else b) for b in base64.b64decode(encoded)]


@dataclass
class PerplexityEmbedder(Embedder):
    id: str = "pplx-embed-v1-4b"
    dimensions: int = 2560
    api_key: Optional[str] = getenv("PERPLEXITY_API_KEY")
    base_url: str = "https://api.perplexity.ai/v1/embeddings"
    headers: Optional[Dict[str, str]] = None
    request_params: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError(
                "API key is required for Perplexity embedder. Set PERPLEXITY_API_KEY environment variable or pass api_key parameter."
            )

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        if self.headers:
            headers.update(self.headers)
        return headers

    def _payload(self, texts: List[str]) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": self.id,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "base64_int8",
        }
        if self.request_params:
            data.update(self.request_params)
        return data

    def _response(self, text: str) -> Dict[str, Any]:
        response = requests.post(
            self.base_url, headers=self._get_headers(), json=self._payload([text]), timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _embedding(self, result: Dict[str, Any]) -> List[float]:
        entry = first_embedding(result.get("data"), "Perplexity")
        return decode_int8_embedding(entry["embedding"]) if entry else []

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self._embedding(self._response(text))
        except Exception as e:
            raise_embedding_error(e, model_id=self.id, provider="Perplexity")

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        try:
            result = self._response(text)
            return self._embedding(result), result.get("usage")
        except Exception as e:
            raise_embedding_error(e, model_id=self.id, provider="Perplexity")

    async def _async_response(self, texts: List[str]) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout) if self.timeout else None

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.base_url, headers=self._get_headers(), json=self._payload(texts)) as response:
                response.raise_for_status()
                return await response.json()

    async def async_get_embedding(self, text: str) -> List[float]:
        try:
            return self._embedding(await self._async_response([text]))
        except Exception as e:
            raise_embedding_error(e, model_id=self.id, provider="Perplexity")

    async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        try:
            result = await self._async_response([text])
            return self._embedding(result), result.get("usage")
        except Exception as e:
            raise_embedding_error(e, model_id=self.id, provider="Perplexity")

    async def async_get_embeddings_batch_and_usage(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Optional[Dict]]]:
        all_embeddings: List[List[float]] = []
        all_usage: List[Optional[Dict]] = []
        logger.info(f"Getting embeddings and usage for {len(texts)} texts in batches of {self.batch_size}")

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            try:
                result = await self._async_response(batch_texts)
                ordered = sorted(result.get("data") or [], key=lambda item: item.get("index", 0))
                batch_embeddings = pad_batch_embeddings(
                    [decode_int8_embedding(item["embedding"]) for item in ordered], batch_texts, "Perplexity"
                )
                all_embeddings.extend(batch_embeddings)

                usage_dict = result.get("usage")
                all_usage.extend([usage_dict] * len(batch_embeddings))
            except Exception as e:
                log_warning(f"Error in async batch embedding: {str(e)}")
                batch_embeddings, batch_usage = await aembed_texts_individually(self, batch_texts)
                all_embeddings.extend(batch_embeddings)
                all_usage.extend(batch_usage)

        return all_embeddings, all_usage
