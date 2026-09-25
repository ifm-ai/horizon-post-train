"""
Utilities for the OpenAI endpoint
"""

import asyncio
import json
import logging
import random
from argparse import Namespace
from copy import copy, deepcopy
from typing import Any

import httpx

from miles.rollout.generate_utils.generate_endpoint_utils import get_rollout_topk_from_response
from miles.rollout.session.session_types import (
    GetMergedSessionResponse,
    GetSessionResponse,
    MergedSessionSample,
    SessionRecord,
)
from miles.utils.types import RolloutSamplingMask, Sample

logger = logging.getLogger(__name__)


def _encoded_routed_experts_bytes(routed_experts) -> int:
    if routed_experts is None:
        return 0
    if isinstance(routed_experts, str):
        return len(routed_experts.encode("ascii"))
    try:
        return len(routed_experts)
    except TypeError:
        return -1


def _expected_routed_experts_bytes(args: Namespace, num_tokens: int) -> int | None:
    num_layers = getattr(args, "num_layers", None)
    moe_router_topk = getattr(args, "moe_router_topk", None)
    if num_layers is None or moe_router_topk is None:
        return None
    return max(num_tokens - 1, 0) * int(num_layers) * int(moe_router_topk) * 4


_SESSION_REQUEST_TIMEOUT = 120.0

_HTTP_CONNECT_TIMEOUT = 10.0
_HTTP_READ_TIMEOUT = 120.0
_HTTP_WRITE_TIMEOUT = 30.0
_HTTP_POOL_TIMEOUT = 10.0

_HEALTH_RETRIES = 2
_CREATE_RETRIES = 10
_COLLECT_RETRIES = 3
_DELETE_RETRIES = 3

_BACKOFF_INITIAL_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 10.0
_BACKOFF_JITTER_FRACTION = 0.2

_COLLECT_RECORDS_CONCURRENCY = 256
_COLLECT_RECORDS_SEMAPHORE = asyncio.Semaphore(_COLLECT_RECORDS_CONCURRENCY)


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
    ):
        self.router_url = router_url.rstrip("/")
        self.session_id = session_id
        self.base_url = f"{self.router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id

    @staticmethod
    def _timeout() -> httpx.Timeout:
        return httpx.Timeout(
            timeout=_SESSION_REQUEST_TIMEOUT,
            connect=_HTTP_CONNECT_TIMEOUT,
            read=_HTTP_READ_TIMEOUT,
            write=_HTTP_WRITE_TIMEOUT,
            pool=_HTTP_POOL_TIMEOUT,
        )

    @staticmethod
    def _response_size(response: httpx.Response) -> int:
        try:
            return len(response.content)
        except Exception:
            return -1

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        # attempt is 1-indexed.
        base_delay = _BACKOFF_INITIAL_SECONDS * (2 ** (attempt - 1))
        base_delay = min(base_delay, _BACKOFF_MAX_SECONDS)

        jitter = random.uniform(
            1.0 - _BACKOFF_JITTER_FRACTION,
            1.0 + _BACKOFF_JITTER_FRACTION,
        )
        return min(base_delay * jitter, _BACKOFF_MAX_SECONDS)

    @classmethod
    async def _request(
        cls,
        method: str,
        url: str,
        *,
        phase: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        expect_json: bool = True,
    ) -> Any:
        method = method.upper()
        last_exc: BaseException | None = None

        async with httpx.AsyncClient(timeout=cls._timeout()) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        "[session-client] request_start phase=%s method=%s url=%s attempt=%d/%d",
                        phase,
                        method,
                        url,
                        attempt,
                        max_retries,
                    )

                    if method in {"GET", "DELETE"}:
                        response = await client.request(method, url, headers=headers)
                    else:
                        response = await client.request(
                            method,
                            url,
                            json=payload or {},
                            headers=headers,
                        )

                    body_bytes = cls._response_size(response)

                    logger.info(
                        "[session-client] response_received phase=%s method=%s url=%s status=%d bytes=%d attempt=%d/%d",
                        phase,
                        method,
                        url,
                        response.status_code,
                        body_bytes,
                        attempt,
                        max_retries,
                    )

                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        last_exc = exc

                        logger.info(
                            "[session-client] http_status_error phase=%s method=%s url=%s status=%d bytes=%d attempt=%d/%d",
                            phase,
                            method,
                            url,
                            status,
                            body_bytes,
                            attempt,
                            max_retries,
                        )

                        if status != 429 and status < 500:
                            raise

                    else:
                        if response.status_code == 204 or not response.content:
                            return None

                        if not expect_json:
                            return response.text

                        try:
                            return response.json()
                        except json.JSONDecodeError as exc:
                            logger.info(
                                "[session-client] json_decode_error phase=%s method=%s url=%s status=%d bytes=%d attempt=%d/%d",
                                phase,
                                method,
                                url,
                                response.status_code,
                                body_bytes,
                                attempt,
                                max_retries,
                            )
                            raise exc

                except asyncio.CancelledError:
                    logger.info(
                        "[session-client] request_cancelled phase=%s method=%s url=%s attempt=%d/%d",
                        phase,
                        method,
                        url,
                        attempt,
                        max_retries,
                    )
                    raise

                except httpx.RemoteProtocolError as exc:
                    last_exc = exc
                    logger.info(
                        "[session-client] remote_protocol_error phase=%s method=%s url=%s attempt=%d/%d error_type=%s error=%r",
                        phase,
                        method,
                        url,
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        exc,
                    )

                except httpx.TimeoutException as exc:
                    last_exc = exc
                    logger.info(
                        "[session-client] timeout phase=%s method=%s url=%s attempt=%d/%d error_type=%s error=%r",
                        phase,
                        method,
                        url,
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        exc,
                    )

                except httpx.TransportError as exc:
                    last_exc = exc
                    logger.info(
                        "[session-client] transport_error phase=%s method=%s url=%s attempt=%d/%d error_type=%s error=%r",
                        phase,
                        method,
                        url,
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        exc,
                    )

                except Exception as exc:
                    logger.info(
                        "[session-client] unexpected_error phase=%s method=%s url=%s attempt=%d/%d error_type=%s error=%r",
                        phase,
                        method,
                        url,
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                if attempt < max_retries:
                    delay = cls._backoff_seconds(attempt)
                    logger.info(
                        "[session-client] request_retry_sleep phase=%s method=%s url=%s attempt=%d/%d sleep_s=%.3f",
                        phase,
                        method,
                        url,
                        attempt,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)

        logger.info(
            "[session-client] request_failed phase=%s method=%s url=%s attempts=%d last_error_type=%s last_error=%r",
            phase,
            method,
            url,
            max_retries,
            type(last_exc).__name__ if last_exc is not None else None,
            last_exc,
        )

        if last_exc is not None:
            raise last_exc

        raise RuntimeError(f"request failed without exception: phase={phase} method={method} url={url}")

    @staticmethod
    async def create(args: Namespace, *, capture_sampling_mask: bool = False):
        backends = getattr(args, "session_server_backends", None)
        if backends:
            session_url = random.choice(backends).rstrip("/")
        else:
            session_ip = getattr(args, "session_server_ip", None)
            session_port = getattr(args, "session_server_port", None)
            if not session_ip or not session_port:
                raise RuntimeError(
                    "session_server_ip/session_server_port are not set. Pass --use-session-server to start the session server."
                )
            session_url = f"http://{session_ip}:{session_port}"

        logger.info("[session-client] create_start session_url=%s", session_url)

        session_server_instance_id = None

        try:
            health = await OpenAIEndpointTracer._request(
                "GET",
                f"{session_url}/health",
                phase="health",
                max_retries=_HEALTH_RETRIES,
            )

            if isinstance(health, dict):
                session_server_instance_id = health.get("session_server_instance_id")
                if session_server_instance_id is not None:
                    args.session_server_instance_id = session_server_instance_id

            logger.info(
                "[session-client] health_ok session_url=%s instance_id=%s",
                session_url,
                session_server_instance_id,
            )

        except Exception as exc:
            logger.info(
                "[session-client] health_failed session_url=%s error_type=%s error=%r",
                session_url,
                type(exc).__name__,
                exc,
            )

        response = await OpenAIEndpointTracer._request(
            "POST",
            f"{session_url}/sessions",
            phase="create_session",
            payload={"capture_sampling_mask": capture_sampling_mask},
            max_retries=_CREATE_RETRIES,
        )

        if not isinstance(response, dict) or "session_id" not in response:
            raise RuntimeError(f"invalid create session response from {session_url}: {response!r}")

        session_id = response["session_id"]

        logger.info(
            "[session-client] create_done session_url=%s session_id=%s instance_id=%s",
            session_url,
            session_id,
            session_server_instance_id,
        )

        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
        )

    async def collect_merged_sample(self) -> tuple[MergedSessionSample | None, dict]:
        merged_url = f"{self.base_url}/merged"
        logger.info(
            "[session-client] collect_merged_wait session_id=%s url=%s concurrency_limit=%d",
            self.session_id,
            merged_url,
            _COLLECT_RECORDS_CONCURRENCY,
        )

        async with _COLLECT_RECORDS_SEMAPHORE:
            logger.info(
                "[session-client] collect_merged_start session_id=%s url=%s",
                self.session_id,
                merged_url,
            )

            try:
                response = await self._request(
                    "GET",
                    merged_url,
                    phase="collect_merged_sample",
                    max_retries=_COLLECT_RETRIES,
                )

                parsed = GetMergedSessionResponse.model_validate(response)
                sample = parsed.sample
                metadata = parsed.metadata or {}

                if sample is None:
                    logger.info(
                        "[session-client] collect_merged_done session_id=%s empty=True metadata_keys=%s",
                        self.session_id,
                        sorted(metadata.keys()),
                    )
                else:
                    logger.info(
                        "[session-client] collect_merged_done session_id=%s empty=False tokens=%d "
                        "response_length=%d loss_mask=%d rollout_log_probs=%d weight_versions=%d "
                        "prefix_cache_meta_infos=%d routed_experts_encoded_bytes=%d status=%s metadata_keys=%s",
                        self.session_id,
                        len(sample.tokens),
                        sample.response_length,
                        len(sample.loss_mask),
                        len(sample.rollout_log_probs),
                        len(sample.weight_versions),
                        len(sample.prefix_cache_meta_infos),
                        _encoded_routed_experts_bytes(sample.rollout_routed_experts),
                        sample.status,
                        sorted(metadata.keys()),
                    )

                return sample, metadata

            except httpx.ConnectTimeout as exc:
                logger.info(
                    "[session-client] collect_merged_failed_connect_timeout session_id=%s url=%s "
                    "connect_timeout_s=%.1f error_type=%s error=%r returning_empty_sample=True",
                    self.session_id,
                    merged_url,
                    _HTTP_CONNECT_TIMEOUT,
                    type(exc).__name__,
                    exc,
                )
                return None, {}

            except httpx.RemoteProtocolError as exc:
                logger.info(
                    "[session-client] collect_merged_failed_remote_protocol session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_sample=True",
                    self.session_id,
                    merged_url,
                    type(exc).__name__,
                    exc,
                )
                return None, {}

            except httpx.TimeoutException as exc:
                logger.info(
                    "[session-client] collect_merged_failed_timeout session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_sample=True",
                    self.session_id,
                    merged_url,
                    type(exc).__name__,
                    exc,
                )
                return None, {}

            except httpx.TransportError as exc:
                logger.info(
                    "[session-client] collect_merged_failed_transport session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_sample=True",
                    self.session_id,
                    merged_url,
                    type(exc).__name__,
                    exc,
                )
                return None, {}

            except Exception as exc:
                logger.info(
                    "[session-client] collect_merged_failed_unexpected session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_sample=True",
                    self.session_id,
                    merged_url,
                    type(exc).__name__,
                    exc,
                )
                return None, {}

            finally:
                await self.delete_session()

    async def collect_records(self) -> tuple[list[SessionRecord], dict]:
        logger.info(
            "[session-client] collect_wait session_id=%s url=%s concurrency_limit=%d",
            self.session_id,
            self.base_url,
            _COLLECT_RECORDS_CONCURRENCY,
        )

        async with _COLLECT_RECORDS_SEMAPHORE:
            logger.info(
                "[session-client] collect_start session_id=%s url=%s",
                self.session_id,
                self.base_url,
            )

            try:
                response = await self._request(
                    "GET",
                    self.base_url,
                    phase="collect_records",
                    max_retries=_COLLECT_RETRIES,
                )

                parsed = GetSessionResponse.model_validate(response)
                records = parsed.records or []
                metadata = parsed.metadata or {}

                logger.info(
                    "[session-client] collect_done session_id=%s records=%d metadata_keys=%s",
                    self.session_id,
                    len(records),
                    sorted(metadata.keys()),
                )

                return records, metadata

            except httpx.ConnectTimeout as exc:
                logger.info(
                    "[session-client] collect_failed_connect_timeout session_id=%s url=%s "
                    "connect_timeout_s=%.1f error_type=%s error=%r returning_empty_records=True",
                    self.session_id,
                    self.base_url,
                    _HTTP_CONNECT_TIMEOUT,
                    type(exc).__name__,
                    exc,
                )
                return [], {}

            except httpx.RemoteProtocolError as exc:
                logger.info(
                    "[session-client] collect_failed_remote_protocol session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_records=True",
                    self.session_id,
                    self.base_url,
                    type(exc).__name__,
                    exc,
                )
                return [], {}

            except httpx.TimeoutException as exc:
                logger.info(
                    "[session-client] collect_failed_timeout session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_records=True",
                    self.session_id,
                    self.base_url,
                    type(exc).__name__,
                    exc,
                )
                return [], {}

            except httpx.TransportError as exc:
                logger.info(
                    "[session-client] collect_failed_transport session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_records=True",
                    self.session_id,
                    self.base_url,
                    type(exc).__name__,
                    exc,
                )
                return [], {}

            except Exception as exc:
                logger.info(
                    "[session-client] collect_failed_unexpected session_id=%s url=%s "
                    "error_type=%s error=%r returning_empty_records=True",
                    self.session_id,
                    self.base_url,
                    type(exc).__name__,
                    exc,
                )
                return [], {}

            finally:
                await self.delete_session()

    async def delete_session(self) -> None:
        logger.info(
            "[session-client] delete_start session_id=%s url=%s",
            self.session_id,
            self.base_url,
        )

        try:
            await self._request(
                "DELETE",
                self.base_url,
                phase="delete_session",
                max_retries=_DELETE_RETRIES,
                expect_json=False,
            )

            logger.info(
                "[session-client] delete_done session_id=%s url=%s",
                self.session_id,
                self.base_url,
            )

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code

            if status == 404:
                logger.info(
                    "[session-client] delete_not_found session_id=%s url=%s",
                    self.session_id,
                    self.base_url,
                )
                return

            logger.info(
                "[session-client] delete_failed_status session_id=%s url=%s status=%d error_type=%s error=%r",
                self.session_id,
                self.base_url,
                status,
                type(exc).__name__,
                exc,
            )

        except Exception as exc:
            logger.info(
                "[session-client] delete_failed session_id=%s url=%s error_type=%s error=%r",
                self.session_id,
                self.base_url,
                type(exc).__name__,
                exc,
            )


def apply_merged_session_sample(
    args: Namespace,
    input_sample: Sample,
    merged: MergedSessionSample,
) -> Sample:
    """Materialize one server-merged session payload into a local Sample."""
    sample = deepcopy(input_sample)
    sample.tokens = merged.tokens
    sample.response = merged.response
    sample.response_length = merged.response_length
    sample.loss_mask = merged.loss_mask
    sample.rollout_log_probs = merged.rollout_log_probs
    sample.rollout_sampling_mask = (
        RolloutSamplingMask.from_dict(merged.rollout_sampling_mask)
        if merged.rollout_sampling_mask is not None
        else None
    )
    sample.metadata = {**(sample.metadata or {}), **(merged.metadata or {})}
    sample.status = Sample.Status(merged.status)
    sample.weight_versions.extend(merged.weight_versions)

    for meta_info in merged.prefix_cache_meta_infos:
        sample.prefix_cache_info.add(meta_info)

    if merged.rollout_routed_experts is not None:
        choice = {"meta_info": {"routed_experts": merged.rollout_routed_experts}}
        sample.rollout_routed_experts = get_rollout_topk_from_response(args, choice, sample, "routed_experts")

    routed_experts_shape = None
    routed_experts_decoded_bytes = 0
    if sample.rollout_routed_experts is not None:
        routed_experts_shape = tuple(sample.rollout_routed_experts.shape)
        routed_experts_decoded_bytes = int(sample.rollout_routed_experts.nbytes)

    expected_routed_experts_bytes = _expected_routed_experts_bytes(args, len(sample.tokens))

    logger.info(
        "[session-client] apply_merged_sample tokens=%d response_length=%d loss_mask=%d "
        "rollout_log_probs=%d weight_versions=%d prefix_cache_meta_infos=%d "
        "routed_experts_encoded_bytes=%d routed_experts_decoded_bytes=%d "
        "expected_routed_experts_bytes=%s routed_experts_shape=%s status=%s",
        len(sample.tokens),
        sample.response_length,
        len(sample.loss_mask or []),
        len(sample.rollout_log_probs or []),
        len(sample.weight_versions),
        len(merged.prefix_cache_meta_infos),
        _encoded_routed_experts_bytes(merged.rollout_routed_experts),
        routed_experts_decoded_bytes,
        expected_routed_experts_bytes,
        routed_experts_shape,
        sample.status.value,
    )

    sample.validate()
    return sample


def compute_samples_from_openai_records(
    args: Namespace,
    input_sample: Sample,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
) -> list[Sample]:
    """Convert per-turn session records into training Samples, aligning each
    turn's output tokens against the TITO accumulated token sequence.

    Each record carries its own ``prompt_token_ids`` and ``output_token_ids``
    (with logprobs).  We want to reuse those per-turn logprobs directly
    instead of re-decoding, but we must first trim "trailing tokens" — stop
    tokens the model emitted that the chat template also renders as the next
    turn's delimiter — to avoid double-counting.

    See ``TestTITOTrailingTokenTrim`` in
    ``tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py``
    for a concrete worked example with token-level walkthroughs.
    """
    samples = []
    cursor = 0

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        prompt_ids = record.response["choices"][0]["prompt_token_ids"]
        output_ids = [t[1] for t in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]

        trim_count = 0
        if accumulated_token_ids is not None:
            cursor = len(prompt_ids)

            matched = 0
            for j in range(len(output_ids)):
                idx = cursor + j
                if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                    matched += 1
                else:
                    break

            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert (
                trim_count <= allowed
            ), f"trim_count {trim_count} exceeds allowed={allowed} (is_last={is_last}, max_trim_tokens={max_trim_tokens}); output_ids[-3:]={output_ids[-3:]}, accumulated[cursor:cursor+3]={accumulated_token_ids[cursor : cursor + 3]}"

            cursor += matched

        sample = _compute_sample_from_openai_record(args, input_sample, record, tokenizer, trim_count)
        samples.append(sample)

    if accumulated_token_ids is not None:
        assert cursor == len(
            accumulated_token_ids
        ), f"cursor {cursor} != len(accumulated_token_ids) {len(accumulated_token_ids)} after processing all {len(records)} records"

    return samples


def _compute_sample_from_openai_record(
    args: Namespace, input_sample: Sample, record: SessionRecord, tokenizer, trim_count: int = 0
) -> Sample:
    choice = record.response["choices"][0]

    if "prompt_token_ids" in choice:
        prompt_token_ids = choice["prompt_token_ids"]
    else:
        raise ValueError("prompt_token_ids not found in response choice — ensure return_prompt_token_ids=True is set")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    rollout_sampling_mask = None
    if record.rollout_sampling_mask is not None:
        rollout_sampling_mask = RolloutSamplingMask.from_dict(record.rollout_sampling_mask)
        output_log_probs = record.rollout_sampling_log_probs

    sample = copy(input_sample)
    sample.metadata = dict(input_sample.metadata)
    sample.weight_versions = list(input_sample.weight_versions)
    sample.prefix_cache_info = deepcopy(input_sample.prefix_cache_info)
    request_input_ids = record.request.get("input_ids")
    if request_input_ids is not None:
        assert (
            request_input_ids == prompt_token_ids
        ), "for prompt part, input_ids return by sglang should match with the request input_ids"

    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.rollout_sampling_mask = rollout_sampling_mask
    sample.response = ""
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    sample.rollout_routed_experts = get_rollout_topk_from_response(args, choice, sample, "routed_experts")

    if not hasattr(sample, "metadata") or sample.metadata is None:
        sample.metadata = {}
    sample.metadata["response_decoded"] = False

    if trim_count > 0:
        _strip_last_output_tokens_without_decode(sample, trim_count)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    sample.prefix_cache_info.add(choice.get("meta_info", {}))
    if "weight_version" in choice["meta_info"]:
        sample.weight_versions.append(choice["meta_info"]["weight_version"])

    return sample


def _strip_last_output_tokens_without_decode(sample: Sample, trim_count: int) -> None:
    """Strip trailing output tokens without decoding sample.response."""
    if trim_count <= 0:
        return

    assert (
        trim_count <= sample.response_length
    ), f"trim_count {trim_count} exceeds response_length {sample.response_length}"

    prompt_len = len(sample.tokens) - sample.response_length
    keep_tokens = sample.response_length - trim_count

    sample.tokens = sample.tokens[: prompt_len + keep_tokens]
    sample.response_length = keep_tokens

    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs = sample.rollout_log_probs[:keep_tokens]
    if sample.rollout_sampling_mask is not None:
        sample.rollout_sampling_mask = sample.rollout_sampling_mask.prefix(keep_tokens)
    if sample.loss_mask is not None:
        sample.loss_mask = sample.loss_mask[:keep_tokens]
    if sample.rollout_routed_experts is not None:
        sample.rollout_routed_experts = sample.rollout_routed_experts[: len(sample.tokens) - 1]


def truncate_samples_by_total_tokens(
    samples: list[Sample],
    max_seq_len: int,
    tokenizer,
) -> list[Sample]:
    """Truncate samples so the total token count (prompt + output, including
    env responses) does not exceed ``max_seq_len``.
    """
    result: list[Sample] = []

    for sample in samples:
        total = len(sample.tokens)
        if total <= max_seq_len:
            result.append(sample)
            continue

        overshoot = total - max_seq_len
        allowed_output = sample.response_length - overshoot
        if allowed_output <= 0:
            break

        _truncate_sample_output(sample, allowed_output, tokenizer)
        result.append(sample)
        break

    return result


def _truncate_sample_output(sample: Sample, keep_tokens: int, tokenizer) -> None:
    """Truncate a sample's output in-place to exactly ``keep_tokens`` tokens."""
    prompt_len = len(sample.tokens) - sample.response_length
    kept_ids = sample.tokens[prompt_len : prompt_len + keep_tokens]

    sample.tokens = sample.tokens[:prompt_len] + kept_ids
    sample.response = ""
    sample.response_length = keep_tokens

    if not hasattr(sample, "metadata") or sample.metadata is None:
        sample.metadata = {}
    sample.metadata["response_decoded"] = False

    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs = sample.rollout_log_probs[:keep_tokens]
    if sample.rollout_sampling_mask is not None:
        sample.rollout_sampling_mask = sample.rollout_sampling_mask.prefix(keep_tokens)
    if sample.loss_mask is not None:
        sample.loss_mask = sample.loss_mask[:keep_tokens]
    if sample.rollout_routed_experts is not None:
        sample.rollout_routed_experts = sample.rollout_routed_experts[: len(sample.tokens) - 1]
    sample.status = Sample.Status.TRUNCATED
