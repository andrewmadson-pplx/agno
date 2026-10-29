import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.exceptions import EmbeddingError
from agno.knowledge.embedder.perplexity import PerplexityEmbedder, decode_int8_embedding


def encode(values):
    return base64.b64encode(bytes(v & 0xFF for v in values)).decode()


def response(*vectors, usage=None):
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": encode(v)} for i, v in enumerate(vectors)],
        "usage": usage,
    }


def test_decode_int8_embedding_is_signed():
    assert decode_int8_embedding(encode([0, 1, 127, -1, -128])) == [0.0, 1.0, 127.0, -1.0, -128.0]


def test_get_embedding_and_usage_sends_int8_request():
    embedder = PerplexityEmbedder(api_key="test", id="pplx-embed-v1-0.6b", dimensions=256)
    http = MagicMock()
    http.json.return_value = response([3, -4], usage={"total_tokens": 2})

    with patch("agno.knowledge.embedder.perplexity.requests.post", return_value=http) as post:
        embedding, usage = embedder.get_embedding_and_usage("hello")

    assert embedding == [3.0, -4.0]
    assert usage == {"total_tokens": 2}
    assert post.call_args.kwargs["json"] == {
        "model": "pplx-embed-v1-0.6b",
        "input": ["hello"],
        "dimensions": 256,
        "encoding_format": "base64_int8",
    }
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test"


def test_missing_api_key_raises_embedding_error():
    with pytest.raises(EmbeddingError):
        PerplexityEmbedder(api_key=None).get_embedding("hello")


@pytest.mark.asyncio
async def test_batch_orders_by_index():
    embedder = PerplexityEmbedder(api_key="test", enable_batch=True, batch_size=10)
    result = response([1], [2])
    result["data"].reverse()
    embedder._async_response = AsyncMock(return_value=result)

    embeddings, usage = await embedder.async_get_embeddings_batch_and_usage(["a", "b"])

    assert embeddings == [[1.0], [2.0]]
    assert len(usage) == 2
