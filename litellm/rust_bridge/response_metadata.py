from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import suppress
from types import MappingProxyType
from typing import Final, Generic, Protocol, TypeVar, cast, runtime_checkable

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.litellm_core_utils.execution import (
    EXECUTION_KEY,
    RUST_HEADER,
    ExecutionOrigin,
    ExecutionResult,
    execution_from_metadata,
    execution_headers,
)
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict

ResultT = TypeVar("ResultT")  # rebind-ok: TypeVar declarations cannot be annotated Final
ChunkT = TypeVar("ChunkT")  # rebind-ok: TypeVar declarations cannot be annotated Final
_HEADERS: Final = TypeAdapter(Mapping[str, object])
_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


class MetadataHost(Protocol):
    _hidden_params: Mapping[str, object]


@runtime_checkable
class AsyncClose(Protocol):
    async def aclose(self) -> None: ...


@runtime_checkable
class SyncClose(Protocol):
    def close(self) -> None: ...


def _headers(value: object) -> Mapping[str, object]:
    try:
        return _HEADERS.validate_python(value, strict=True)
    except ValidationError:
        return MappingProxyType({})


def get_execution(response: object) -> ExecutionOrigin:
    if isinstance(response, ExecutionResult):
        return response.origin
    hidden: Final = get_hidden_params_dict(response)
    origin: Final = execution_from_metadata(hidden)
    if origin is not ExecutionOrigin.NOT_EXECUTED:
        return origin
    attached: Final[object] = getattr(response, EXECUTION_KEY, None)
    return attached if isinstance(attached, ExecutionOrigin) else ExecutionOrigin.NOT_EXECUTED


def mark_execution_error(error: BaseException, origin: ExecutionOrigin) -> None:
    setattr(error, EXECUTION_KEY, origin)


def execution_metadata(
    metadata: Mapping[str, object], origin: ExecutionOrigin
) -> dict[str, object]:  # mutable-ok: SDK and router mutate _hidden_params in place
    headers: Final = _headers(metadata.get("additional_headers"))
    additional: Final = dict(execution_headers(headers, origin))  # mutable-ok: routing helpers update these headers
    return dict(  # mutable-ok: SDK and router mutate _hidden_params in place
        (*metadata.items(), (EXECUTION_KEY, origin), ("additional_headers", additional))
    )


def _attach_metadata(response: object, origin: ExecutionOrigin) -> None:
    host: Final = cast(MetadataHost, response)  # cast-ok: callers check the dynamically exposed SDK attribute
    host._hidden_params = (  # pyright: ignore[reportPrivateUsage]  # SDK metadata compatibility
        execution_metadata(get_hidden_params_dict(response), origin)
    )


def expose_result(result: ExecutionResult[ResultT]) -> ResultT:
    response: Final[object] = result.value
    origin: Final = ExecutionOrigin.CACHE if get_execution(response) is ExecutionOrigin.CACHE else result.origin
    if origin is ExecutionOrigin.CACHE and hasattr(response, "_hidden_params"):
        _attach_metadata(response, origin)
        return result.value
    if isinstance(response, AsyncIterator):
        if isinstance(response, ExecutionAsyncStream) and get_execution(result.value) is origin:
            return result.value
        stream: Final = cast(  # cast-ok: isinstance established the async iterator protocol
            AsyncIterator[object], response
        )
        return cast(ResultT, ExecutionAsyncStream(stream, origin))  # cast-ok: preserves the async iterator contract
    if isinstance(response, Iterator):
        if isinstance(response, ExecutionStream) and get_execution(result.value) is origin:
            return result.value
        sync_stream: Final = cast(Iterator[object], response)  # cast-ok: isinstance established the iterator protocol
        return cast(ResultT, ExecutionStream(sync_stream, origin))  # cast-ok: preserves the iterator contract
    if isinstance(response, dict):
        response["_hidden_params"] = execution_metadata(get_hidden_params_dict(result.value), origin)
        return result.value
    if hasattr(response, "_hidden_params"):
        _attach_metadata(response, origin)
        return result.value
    if isinstance(response, (BaseModel, BaseException, httpx.Response)):
        setattr(response, EXECUTION_KEY, origin)
        if isinstance(response, httpx.Response):
            if RUST_HEADER in response.headers:
                del response.headers[RUST_HEADER]
            response.headers.update(execution_headers(_NO_HEADERS, origin))
        return result.value
    if origin is ExecutionOrigin.RUST and response is not None:
        error: Final = TypeError(f"No execution metadata adapter for {type(response).__name__}; use run_result")
        mark_execution_error(error, origin)
        raise error
    return result.value


class ExecutionStream(Generic[ChunkT]):
    def __init__(self, inner: Iterator[ChunkT], origin: ExecutionOrigin) -> None:
        self._inner: Final = inner
        self._hidden_params = execution_metadata(get_hidden_params_dict(inner), origin)
        self._closed = False

    def __getattr__(self, name: str) -> object:
        value: Final = cast(  # cast-ok: dynamic delegation preserves provider attributes
            object, getattr(self._inner, name)
        )
        return value

    def __iter__(self) -> ExecutionStream[ChunkT]:
        return self

    def __next__(self) -> ChunkT:
        if self._closed:
            raise StopIteration
        try:
            return next(self._inner)
        except StopIteration:
            self.close()
            raise
        except BaseException as error:
            mark_execution_error(error, get_execution(self))
            with suppress(BaseException):
                self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self._inner, SyncClose):
            self._inner.close()


class ExecutionAsyncStream(Generic[ChunkT]):
    def __init__(self, inner: AsyncIterator[ChunkT], origin: ExecutionOrigin) -> None:
        self._inner: Final = inner
        self._hidden_params = execution_metadata(get_hidden_params_dict(inner), origin)
        self._closed = False

    def __getattr__(self, name: str) -> object:
        value: Final = cast(  # cast-ok: dynamic delegation preserves provider attributes
            object, getattr(self._inner, name)
        )
        return value

    def __aiter__(self) -> ExecutionAsyncStream[ChunkT]:
        return self

    async def __anext__(self) -> ChunkT:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await anext(self._inner)
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException as error:
            mark_execution_error(error, get_execution(self))
            with suppress(BaseException):
                await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self._inner, AsyncClose):
            await self._inner.aclose()


def mark_rust_response(response: ResultT) -> ResultT:
    return expose_result(ExecutionResult(response, ExecutionOrigin.RUST))
