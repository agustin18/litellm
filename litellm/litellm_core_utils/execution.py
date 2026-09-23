from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Generic, TypeVar


class ExecutionOrigin(str, Enum):
    RUST = "rust"
    PYTHON = "python"
    CACHE = "cache"
    NOT_EXECUTED = "not_executed"


EXECUTION_KEY: Final = "_litellm_execution"
RUST_HEADER: Final = "x-litellm-rust"
ValueT = TypeVar("ValueT")  # rebind-ok: TypeVar declarations cannot be annotated Final
MappedT = TypeVar("MappedT")  # rebind-ok: TypeVar declarations cannot be annotated Final
HeaderT = TypeVar("HeaderT")  # rebind-ok: TypeVar declarations cannot be annotated Final


@dataclass(frozen=True, slots=True)
class ExecutionResult(Generic[ValueT]):
    value: ValueT
    origin: ExecutionOrigin

    def map(self, transform: Callable[[ValueT], MappedT]) -> ExecutionResult[MappedT]:
        return ExecutionResult(transform(self.value), self.origin)


def execution_from_metadata(metadata: Mapping[str, object]) -> ExecutionOrigin:
    if metadata.get("cache_hit") is True:
        return ExecutionOrigin.CACHE
    origin: Final = metadata.get(EXECUTION_KEY)
    return origin if isinstance(origin, ExecutionOrigin) else ExecutionOrigin.NOT_EXECUTED


def execution_headers(headers: Mapping[str, HeaderT], origin: ExecutionOrigin) -> Mapping[str, HeaderT | str]:
    return MappingProxyType(
        {
            key: value
            for key, value in (
                *((key, value) for key, value in headers.items() if key.lower() != RUST_HEADER),
                *(((RUST_HEADER, "true"),) if origin is ExecutionOrigin.RUST else ()),
            )
        }
    )
