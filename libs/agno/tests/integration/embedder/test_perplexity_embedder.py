import pytest

from agno.knowledge.embedder.perplexity import PerplexityEmbedder


@pytest.fixture
def embedder():
    return PerplexityEmbedder()


def test_embedder_initialization(embedder):
    assert embedder.id == "pplx-embed-v1-4b"
    assert embedder.dimensions == 2560
    assert embedder.api_key is not None, "PERPLEXITY_API_KEY env variable is not set"


def test_get_embedding_and_usage(embedder):
    embedding, usage = embedder.get_embedding_and_usage("The quick brown fox jumps over the lazy dog.")
    assert len(embedding) == embedder.dimensions
    assert all(isinstance(x, float) for x in embedding)
    assert usage is None or isinstance(usage, dict)


def test_custom_dimensions():
    embedder = PerplexityEmbedder(id="pplx-embed-v1-0.6b", dimensions=256)
    assert len(embedder.get_embedding("Hello, world! こんにちは 123 @#$%")) == 256


@pytest.mark.asyncio
async def test_async_batch(embedder):
    embedder.enable_batch = True
    texts = ["First text", "Second text", "Third text"]
    embeddings, usage = await embedder.async_get_embeddings_batch_and_usage(texts)
    assert len(embeddings) == len(texts)
    assert all(len(e) == embedder.dimensions for e in embeddings)
    assert len(usage) == len(texts)
