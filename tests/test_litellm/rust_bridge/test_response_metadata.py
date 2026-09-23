import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx
import pytest
from pydantic import BaseModel

from litellm.litellm_core_utils.execution import EXECUTION_KEY, ExecutionOrigin, ExecutionResult, execution_headers
from litellm.router import FallbackAwareAnthropicMessagesStream
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge.response_metadata import expose_result, get_execution, mark_rust_response


class Payload(BaseModel):
    text: str


@pytest.mark.parametrize("response", ({"text": "done"}, Payload(text="done"), httpx.Response(200, text="done")))
def test_completed_results_expose_execution_without_changing_payload_type(response: object) -> None:
    marked: Final = mark_rust_response(response)
    assert marked is response
    assert get_execution(marked) is ExecutionOrigin.RUST
    if isinstance(marked, Payload):
        assert marked.model_dump() == {"text": "done"}
    if isinstance(marked, httpx.Response):
        assert marked.text == "done"
        assert marked.headers["x-litellm-rust"] == "true"


def test_result_replacement_retains_provenance_and_reserved_header_cannot_be_overridden() -> None:
    native: Final = ExecutionResult(b"original", ExecutionOrigin.RUST)
    replacement: Final = native.map(lambda _: b"modified")
    assert replacement.value == b"modified"
    assert execution_headers({"X-LiteLLM-Rust": "false", "x-request-id": "kept"}, replacement.origin) == {
        "x-request-id": "kept",
        "x-litellm-rust": "true",
    }
    assert execution_headers({"X-LiteLLM-Rust": "true"}, ExecutionOrigin.PYTHON) == {}


def test_cached_or_deserialized_provenance_cannot_claim_current_native_execution() -> None:
    native: Final = mark_rust_response({"text": "done"})
    cached: Final = expose_result(ExecutionResult(native, ExecutionOrigin.CACHE))
    assert get_execution(cached) is ExecutionOrigin.CACHE
    assert get_hidden_params_dict(cached)["additional_headers"] == {}
    decoded: Final = json.loads(json.dumps(mark_rust_response({"text": "done"})))
    assert get_execution(decoded) is ExecutionOrigin.NOT_EXECUTED


class Source:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure: Final = failure
        self.reads = 0
        self.closes = 0

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        self.reads += 1
        if self.failure is not None:
            raise self.failure
        if self.reads == 1:
            return b"event: message_start\ndata: {}\n\n"
        raise StopIteration

    def close(self) -> None:
        self.closes += 1


class AsyncSource:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.source: Final = Source(failure)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self.source)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.source.close()


def test_sync_stream_is_lazy_preserves_bytes_and_closes_once() -> None:
    source: Final = Source()
    stream: Final = mark_rust_response(source)
    assert source.reads == 0
    assert get_execution(stream) is ExecutionOrigin.RUST
    assert tuple(stream) == (b"event: message_start\ndata: {}\n\n",)
    stream.close()
    assert source.closes == 1


@pytest.mark.asyncio
async def test_async_stream_is_lazy_preserves_bytes_and_closes_once() -> None:
    source: Final = AsyncSource()
    stream: Final = mark_rust_response(source)
    assert source.source.reads == 0
    assert get_execution(stream) is ExecutionOrigin.RUST
    assert tuple([chunk async for chunk in stream]) == (b"event: message_start\ndata: {}\n\n",)
    await stream.aclose()
    assert source.source.closes == 1


@pytest.mark.parametrize("failure", (ValueError("upstream failed"), asyncio.CancelledError()))
def test_sync_stream_failure_keeps_exception_identity_and_closes(failure: BaseException) -> None:
    source: Final = Source(failure)
    stream: Final = mark_rust_response(source)
    with pytest.raises(type(failure)) as caught:
        next(stream)
    assert caught.value is failure
    assert get_execution(failure) is ExecutionOrigin.RUST
    assert source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (ValueError("upstream failed"), asyncio.CancelledError()))
async def test_async_stream_failure_keeps_exception_identity_and_closes(failure: BaseException) -> None:
    source: Final = AsyncSource(failure)
    stream: Final = mark_rust_response(source)
    with pytest.raises(type(failure)) as caught:
        await anext(stream)
    assert caught.value is failure
    assert get_execution(failure) is ExecutionOrigin.RUST
    assert source.source.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial,selected", ((ExecutionOrigin.RUST, ExecutionOrigin.PYTHON), (ExecutionOrigin.PYTHON, ExecutionOrigin.RUST))
)
async def test_messages_fallback_replaces_execution_without_rewriting_committed_headers(
    initial: ExecutionOrigin, selected: ExecutionOrigin
) -> None:
    first: Final = expose_result(ExecutionResult(AsyncSource(), initial))
    fallback: Final = expose_result(ExecutionResult(AsyncSource(), selected))

    async def chunks() -> AsyncIterator[bytes]:
        async for chunk in fallback:
            yield chunk

    stream: Final = FallbackAwareAnthropicMessagesStream(chunks(), first)
    committed_headers: Final = execution_headers({}, get_execution(stream))
    stream.adopt_fallback_source(fallback)
    stream.merge_fallback_hidden_params(get_hidden_params_dict(fallback), {})
    assert get_execution(stream) is selected
    assert get_hidden_params_dict(stream)["additional_headers"] == execution_headers({}, selected)
    assert committed_headers == execution_headers({}, initial)
    assert tuple([chunk async for chunk in stream]) == (b"event: message_start\ndata: {}\n\n",)
    await first.aclose()


def test_cache_hit_overrides_live_origin_in_memory() -> None:
    response: Final = {"_hidden_params": {EXECUTION_KEY: ExecutionOrigin.RUST, "cache_hit": True}}
    assert get_execution(response) is ExecutionOrigin.CACHE


def test_native_value_without_response_adapter_requires_explicit_result_envelope() -> None:
    with pytest.raises(TypeError, match="No execution metadata adapter for bytes") as caught:
        mark_rust_response(b"binary")
    assert get_execution(caught.value) is ExecutionOrigin.RUST
    result: Final = ExecutionResult(b"binary", ExecutionOrigin.RUST)
    assert result.value == b"binary"
    assert get_execution(result) is ExecutionOrigin.RUST
