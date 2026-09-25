"""gRPC utility functions."""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from http import HTTPStatus

import grpc


def prefill_prealloc_queue_reqs(disaggregation: object) -> int:
    """Read the prefill pre-allocation queue across SGLang snapshot versions.

    SGLang 0.5.16 publishes this counter as
    ``prefill_bootstrap_queue_reqs``. Newer snapshots use the protobuf-aligned
    ``prefill_prealloc_queue_reqs`` name. Prefer the new name while preserving
    compatibility with the oldest SGLang release supported by this package.
    """
    return getattr(
        disaggregation,
        "prefill_prealloc_queue_reqs",
        getattr(disaggregation, "prefill_bootstrap_queue_reqs", 0),
    )


def to_token_id_array(token_ids: Iterable[int] | None) -> array | None:
    """Coerce a token-id sequence to the ``array("q")`` the SGLang scheduler expects.

    SGLang declares ``TokenizedGenerateReqInput.input_ids`` /
    ``TokenizedEmbeddingReqInput.input_ids`` as ``Optional[array[int]]`` and its
    ``Req`` concatenates ``origin_input_ids + output_ids`` where ``output_ids``
    is ``array("q")``; a plain ``list`` (as gRPC repeated fields decode to) makes
    that concatenation raise ``TypeError``. ``array("q", x)`` accepts any
    iterable of ints, so this is safe at every call site. Returns ``None`` for
    ``None`` input.
    """
    if token_ids is None:
        return None
    return array("q", token_ids)


_HTTP_TO_GRPC_CODE = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
}


def abort_code_from_output(output: dict) -> grpc.StatusCode:
    """Map a scheduler error output to the appropriate gRPC status code."""
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        status_code = finish_reason.get("status_code")
        if status_code is not None:
            return _HTTP_TO_GRPC_CODE.get(status_code, grpc.StatusCode.INTERNAL)
    return grpc.StatusCode.INTERNAL
