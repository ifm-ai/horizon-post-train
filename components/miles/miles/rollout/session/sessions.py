import asyncio
import logging
import time
import uuid

import orjson
import pybase64
from fastapi import Body, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_errors import SessionError, SessionNotFoundError, UpstreamResponseError
from miles.rollout.session.session_types import (
    GetMergedSessionResponse,
    GetSessionResponse,
    MergedSessionSample,
    SessionRecord,
)
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import RolloutSamplingMask

logger = logging.getLogger(__name__)


# Per-process observability counters. Populated by ``setup_session_routes`` and
# read by the background stats logger in ``session_server.run_session_server``.
# Keyed by ``session_server_port`` so multi-backend (N>1) setups don't clobber
# each other when sharing a Python process (currently they don't, but cheap
# insurance). The "default" key is used when ``session_server_port`` is unset.
_worker_stats: dict = {}


def _compact_sampling_replay(meta_info: dict) -> tuple[dict[str, str], list[float]]:
    try:
        masks = meta_info.pop("output_token_sampling_mask")
        log_probs = meta_info.pop("output_token_sampling_logprobs")
        if not (len(masks) == len(log_probs) == meta_info["completion_tokens"] and all(masks)):
            raise ValueError("expected one nonempty mask and logprob per output token")
        mask = RolloutSamplingMask.from_rows(masks).to_dict()
        meta_info.pop("output_token_sampling_mask_length", None)
        return mask, [float(value) for value in log_probs]
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise UpstreamResponseError(f"invalid sampling-mask metadata: {exc}") from exc


def get_worker_stats(port):
    """Return the live stats dict for the given worker port.

    Returns ``None`` if the routes haven't been wired up yet (e.g. health
    probe before first request handler initialisation).
    """
    return _worker_stats.get(port if port is not None else "default")


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    session_server_instance_id = getattr(args, "session_server_instance_id", None)

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        allowed_append_roles=getattr(args, "tito_allowed_append_roles", None),
    )

    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)

    @app.get("/health")
    async def health():
        body = {"status": "ok"}
        if session_server_instance_id is not None:
            body["session_server_instance_id"] = session_server_instance_id
        return body

    # In-flight chat_completions counter (read by the stats logger).
    _inflight_chat = {"count": 0}

    # Per-worker observability counters read by the background stats logger.
    # reqs_total counts handler entries; turns_completed counts turns whose
    # state update succeeded.
    worker_port = getattr(args, "session_server_port", None)
    _stats = {
        "reqs_total": 0,
        "turns_completed": 0,
        "inflight": _inflight_chat,
    }
    _worker_stats[worker_port if worker_port is not None else "default"] = _stats

    @app.middleware("http")
    async def debug_request_logger(request: Request, call_next):
        client = request.client
        client_info = f"{client.host}:{client.port}" if client else "unknown"
        logger.debug(
            f"[session-server] REQUEST ARRIVED: {request.method} {request.url.path} from={client_info} inflight_chat={_inflight_chat['count']}"
        )
        t0 = time.time()
        response = await call_next(request)
        elapsed = time.time() - t0
        logger.debug(
            f"[session-server] REQUEST DONE: {request.method} {request.url.path} status={response.status_code} elapsed={elapsed:.3f}s from={client_info}"
        )
        return response

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.post("/sessions")
    async def create_session(capture_sampling_mask: bool = Body(False, embed=True)):
        session_id = registry.create_session(capture_sampling_mask=capture_sampling_mask)
        return {"session_id": session_id}

    def _compact_output_token_logprobs(output_token_logprobs):
        """Keep only the fields consumed by the sample builder.

        Consumer uses:
          output_log_probs = [item[0] for item in output_token_logprobs]
          output_token_ids = [item[1] for item in output_token_logprobs]

        Some backends may include extra per-token fields; do not return them.
        """
        if not output_token_logprobs:
            return []

        compact = []
        for item in output_token_logprobs:
            compact.append([item[0], item[1]])
        return compact

    def _compact_session_record(record: SessionRecord) -> SessionRecord:
        """Return only the record fields required by compute_samples_from_openai_records."""
        choice = record.response.get("choices", [{}])[0]
        meta_info = choice.get("meta_info") or {}

        compact_meta_info = {
            "output_token_logprobs": _compact_output_token_logprobs(meta_info.get("output_token_logprobs")),
        }

        # Used by _compute_sample_from_openai_record.
        if "weight_version" in meta_info:
            compact_meta_info["weight_version"] = meta_info["weight_version"]

        # Preserve prefix-cache metadata if present. The consumer does:
        # sample.prefix_cache_info.add(choice.get("meta_info", {}))
        for key in (
            "cached_tokens",
            "prompt_tokens",
            "prefix_cache_hit",
            "prefix_cache_len",
            "prefix_cache_tokens",
            "prefix_cache_start_len",
            "prefix_cache_end_len",
        ):
            if key in meta_info:
                compact_meta_info[key] = meta_info[key]

        # Preserve routed experts if present. get_rollout_topk_from_response()
        # may read this from choice or meta_info depending on implementation.
        if "routed_experts" in meta_info:
            compact_meta_info["routed_experts"] = meta_info["routed_experts"]

        compact_choice = {
            "prompt_token_ids": choice.get("prompt_token_ids", []),
            "finish_reason": choice.get("finish_reason"),
            "meta_info": compact_meta_info,
        }

        if "routed_experts" in choice:
            compact_choice["routed_experts"] = choice["routed_experts"]

        return SessionRecord(
            timestamp=record.timestamp,
            method=record.method,
            path=record.path,
            status_code=record.status_code,
            request={
                # Only used for this assertion:
                # request_input_ids == prompt_token_ids
                "input_ids": record.request.get("input_ids"),
            },
            response={
                "choices": [compact_choice],
            },
            rollout_sampling_mask=record.rollout_sampling_mask,
            rollout_sampling_log_probs=record.rollout_sampling_log_probs,
        )

    def _status_from_finish_reason(finish_reason: str | None) -> str:
        match finish_reason:
            case "stop" | "tool_calls":
                return "completed"
            case "length":
                return "truncated"
            case "abort":
                return "aborted"
            case _:
                return "completed"

    def _keep_records_until_first_non_completed(records: list[SessionRecord]) -> tuple[list[SessionRecord], int]:
        """Match sample_utils.drop_samples_after_first_non_completed semantics.

        Keep records through the first non-completed turn and drop any later
        records, because later turns were conditioned on incomplete output.
        """
        for i, record in enumerate(records[:-1]):
            choice = record.response.get("choices", [{}])[0]
            status = _status_from_finish_reason(choice.get("finish_reason"))
            if status != "completed":
                return records[: i + 1], len(records) - i - 1
        return records, 0

    def _routed_experts_encoded_bytes(routed_experts) -> int:
        if routed_experts is None:
            return 0
        if isinstance(routed_experts, str):
            return len(routed_experts.encode("ascii"))
        try:
            return len(routed_experts)
        except TypeError:
            return -1

    def _routed_experts_decoded_bytes(routed_experts) -> int:
        if not isinstance(routed_experts, str):
            return 0
        try:
            return len(pybase64.b64decode(routed_experts.encode("ascii")))
        except Exception:
            return -1

    def _expected_routed_experts_bytes(num_tokens: int) -> int | None:
        num_layers = getattr(args, "num_layers", None)
        moe_router_topk = getattr(args, "moe_router_topk", None)
        if num_layers is None or moe_router_topk is None:
            return None
        return max(num_tokens - 1, 0) * int(num_layers) * int(moe_router_topk) * 4

    def _merge_session_records_to_sample(
        records: list[SessionRecord],
        accumulated_token_ids: list[int],
        max_trim_tokens: int,
        capture_sampling_mask: bool,
    ) -> MergedSessionSample | None:
        if not records:
            return None

        assert accumulated_token_ids, "cannot merge records without accumulated_token_ids"

        tokens = accumulated_token_ids
        first_prompt_len: int | None = None
        loss_mask_full = [0] * len(tokens)
        rollout_log_probs_full = [0.0] * len(tokens)
        sampling_mask_parts = []
        weight_versions: list[str] = []
        prefix_cache_meta_infos: list[dict] = []
        routed_experts = None
        prev_checkpoint: list[int] | None = None
        final_cursor = 0
        final_status = "completed"

        for i, record in enumerate(records):
            is_last = i == len(records) - 1
            assert "choices" in record.response and record.response["choices"], f"record {i} missing response.choices"
            choice = record.response["choices"][0]
            assert "prompt_token_ids" in choice, f"record {i} missing choice.prompt_token_ids"
            prompt_ids = choice["prompt_token_ids"]
            assert isinstance(prompt_ids, list), f"record {i} prompt_token_ids must be a list"

            request_input_ids = record.request.get("input_ids")
            if request_input_ids is not None:
                assert request_input_ids == prompt_ids, f"record {i}: request.input_ids must equal prompt_token_ids"

            meta_info = choice.get("meta_info")
            assert isinstance(meta_info, dict), f"record {i} missing choice.meta_info"
            output_pairs = meta_info.get("output_token_logprobs")
            assert isinstance(output_pairs, list), f"record {i} missing meta_info.output_token_logprobs"

            for j, item in enumerate(output_pairs):
                assert (
                    isinstance(item, (list, tuple)) and len(item) >= 2
                ), f"record {i} output_token_logprobs[{j}] must contain [logprob, token_id]"
                assert isinstance(item[1], int), f"record {i} output_token_logprobs[{j}][1] must be token_id int"

            output_ids = [item[1] for item in output_pairs]
            output_log_probs = [item[0] for item in output_pairs]

            if first_prompt_len is None:
                first_prompt_len = len(prompt_ids)

            if prev_checkpoint is not None:
                obs_len = len(prompt_ids) - len(prev_checkpoint)
                assert obs_len > 0, f"record {i}: obs_len must be > 0, got {obs_len}"
                check_len = max(0, len(prev_checkpoint) - max_trim_tokens)
                assert prompt_ids[:check_len] == prev_checkpoint[:check_len], (
                    f"record {i}: prompt_ids are not compatible with previous checkpoint "
                    f"within max_trim_tokens={max_trim_tokens}"
                )

            cursor = len(prompt_ids)
            matched = 0
            for j, output_id in enumerate(output_ids):
                idx = cursor + j
                if idx < len(tokens) and output_id == tokens[idx]:
                    matched += 1
                else:
                    break

            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert trim_count <= allowed, (
                f"record {i}: trim_count={trim_count} exceeds allowed={allowed}; "
                f"is_last={is_last}, max_trim_tokens={max_trim_tokens}"
            )

            if capture_sampling_mask:
                mask = RolloutSamplingMask.from_dict(record.rollout_sampling_mask)
                obs_start = len(prev_checkpoint) if prev_checkpoint is not None else first_prompt_len
                sampling_mask_parts.extend(
                    (
                        RolloutSamplingMask.singletons(tokens[obs_start:cursor]),
                        mask.prefix(matched) if trim_count else mask,
                    )
                )
                output_log_probs = record.rollout_sampling_log_probs

            for j in range(matched):
                idx = cursor + j
                assert idx < len(tokens), f"record {i}: output token index {idx} outside accumulated tokens"
                loss_mask_full[idx] = 1
                rollout_log_probs_full[idx] = output_log_probs[j]

            if "weight_version" in meta_info:
                weight_versions.append(str(meta_info["weight_version"]))

            prefix_cache_meta_infos.append(
                {key: meta_info[key] for key in ("cached_tokens", "prompt_tokens") if key in meta_info}
            )

            routed_experts = meta_info.get("routed_experts", choice.get("routed_experts"))
            final_status = _status_from_finish_reason(choice.get("finish_reason"))
            if not is_last:
                assert (
                    final_status == "completed"
                ), f"record {i}: only the final record may be non-completed, got {final_status}"

            prev_checkpoint = prompt_ids + output_ids[:matched]
            final_cursor = max(final_cursor, len(prev_checkpoint))

            logger.debug(
                "[session-server] merged_record record_index=%d prompt_tokens=%d output_tokens=%d "
                "matched_output_tokens=%d trim_count=%d routed_experts_encoded_bytes=%d "
                "routed_experts_decoded_bytes=%d",
                i,
                len(prompt_ids),
                len(output_ids),
                matched,
                trim_count,
                _routed_experts_encoded_bytes(routed_experts),
                _routed_experts_decoded_bytes(routed_experts),
            )

        assert final_cursor == len(tokens), f"merged cursor {final_cursor} != accumulated length {len(tokens)}"
        assert first_prompt_len is not None, "first_prompt_len was not initialized"

        response_length = len(tokens) - first_prompt_len
        assert response_length >= 0, f"response_length must be non-negative, got {response_length}"
        assert len(loss_mask_full) == len(tokens), "loss_mask_full length must equal tokens length"
        assert len(rollout_log_probs_full) == len(tokens), "rollout_log_probs_full length must equal tokens length"
        assert len(loss_mask_full[first_prompt_len:]) == response_length, "loss_mask length must equal response_length"
        assert (
            len(rollout_log_probs_full[first_prompt_len:]) == response_length
        ), "rollout_log_probs length must equal response_length"

        routed_experts_decoded_bytes = _routed_experts_decoded_bytes(routed_experts)
        expected_routed_experts_bytes = _expected_routed_experts_bytes(len(tokens))
        if routed_experts is not None and expected_routed_experts_bytes is not None:
            assert routed_experts_decoded_bytes == expected_routed_experts_bytes, (
                f"routed_experts decoded bytes {routed_experts_decoded_bytes} != "
                f"expected {expected_routed_experts_bytes}"
            )

        return MergedSessionSample(
            tokens=tokens,
            response="",
            response_length=response_length,
            loss_mask=loss_mask_full[first_prompt_len:],
            rollout_log_probs=rollout_log_probs_full[first_prompt_len:],
            rollout_sampling_mask=(
                RolloutSamplingMask.concatenate(*sampling_mask_parts).to_dict() if capture_sampling_mask else None
            ),
            status=final_status,
            metadata={"response_decoded": False},
            weight_versions=weight_versions,
            prefix_cache_meta_infos=prefix_cache_meta_infos,
            rollout_routed_experts=routed_experts,
        )

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = registry.sessions.get(session_id)
        if session is None:
            if registry.is_deleted(session_id):
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return GetSessionResponse(session_id=session_id, records=[], metadata={})

        metadata = {
            "accumulated_token_ids": session.token_ids,
            "max_trim_tokens": registry.tito_tokenizer.max_trim_tokens,
        }

        return GetSessionResponse(
            session_id=session_id,
            records=[_compact_session_record(record) for record in session.records],
            metadata=metadata,
        )

    @app.get("/sessions/{session_id}/merged")
    async def get_merged_session(session_id: str):
        session = registry.sessions.get(session_id)
        if session is None:
            if registry.is_deleted(session_id):
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return GetMergedSessionResponse(session_id=session_id, sample=None, metadata={})

        async with session.lock:
            records = [_compact_session_record(record) for record in session.records]
            records_to_merge, dropped_records = _keep_records_until_first_non_completed(records)
            if records_to_merge:
                # Records and trajectory_token_ids are kept in lockstep by
                # LinearTrajectory.append_record/update_pretokenized_state and
                # rollback truncates both together.  If we drop trailing records,
                # use the checkpoint for the last kept record so the merged
                # tokens match the old rollout-manager merge result.
                accumulated_token_ids = list(session.trajectory_token_ids[len(records_to_merge) - 1])
            else:
                accumulated_token_ids = []
            max_trim_tokens = registry.tito_tokenizer.max_trim_tokens

        sample = await asyncio.to_thread(
            _merge_session_records_to_sample,
            records_to_merge,
            accumulated_token_ids,
            max_trim_tokens,
            session.capture_sampling_mask,
        )
        metadata = {
            "records_total": len(records),
            "records_merged": len(records_to_merge),
            "records_dropped_after_first_non_completed": dropped_records,
            "accumulated_token_count": len(accumulated_token_ids),
            "max_trim_tokens": max_trim_tokens,
        }

        if sample is None:
            logger.info(
                "[session-server] merged_collect_empty session_id=%s records=%d accumulated_tokens=%d",
                session_id,
                len(records_to_merge),
                len(accumulated_token_ids),
            )
        else:
            expected_routed_experts_bytes = _expected_routed_experts_bytes(len(sample.tokens))
            logger.info(
                "[session-server] merged_collect_done session_id=%s records=%d tokens=%d response_length=%d "
                "loss_mask=%d rollout_log_probs=%d weight_versions=%d prefix_cache_meta_infos=%d "
                "routed_experts_encoded_bytes=%d routed_experts_decoded_bytes=%d "
                "expected_routed_experts_bytes=%s status=%s",
                session_id,
                len(records_to_merge),
                len(sample.tokens),
                sample.response_length,
                len(sample.loss_mask),
                len(sample.rollout_log_probs),
                len(sample.weight_versions),
                len(sample.prefix_cache_meta_infos),
                _routed_experts_encoded_bytes(sample.rollout_routed_experts),
                _routed_experts_decoded_bytes(sample.rollout_routed_experts),
                expected_routed_experts_bytes,
                sample.status,
            )

        return GetMergedSessionResponse(session_id=session_id, sample=sample, metadata=metadata)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        logger.debug(
            f"[session-server] DELETE waiting for lock: session={session_id} lock_locked={session.lock.locked()}"
        )
        await session.lock.acquire()
        logger.debug(f"[session-server] DELETE acquired lock: session={session_id}")
        try:
            registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        """Proxy a chat completion through SGLang with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → inject
        SGLang flags → proxy to backend (NO lock) → validate response →
        update trajectory checkpoint (lock held briefly) → append session record.

        The lock is NOT held during the slow proxy call to avoid blocking
        DELETE/other operations when the agent disconnects mid-request.
        """
        # Observability state for the chat_done log: t_handler_start anchors
        # total_ms; the lock_wait / tokenize_in / proxy / tokenize_out spans are
        # seeded equal to it so the log emits 0.0 ms for any span the handler
        # skipped (e.g. early 404 / 410 / upstream error short-circuits).
        t_handler_start = time.monotonic()
        t_lock_wait_start = t_handler_start
        t_lock_acquired = t_handler_start
        t_tok_in_start = t_handler_start
        t_tok_in_end = t_handler_start
        t_proxy_start = t_handler_start
        t_proxy_end = t_handler_start
        t_tok_out_start = t_handler_start
        t_tok_out_end = t_handler_start
        prompt_tokens_emit = -1
        completion_tokens_emit = -1
        req_id = uuid.uuid4().hex[:8]

        # Read body up-front so chat_start can log messages_len. Starlette
        # consumes the body at most once; cache it for reuse under the lock
        # below. Parsing it outside the lock trims lock-hold time.
        _raw_body = await request.body()
        try:
            _early_request_body = orjson.loads(_raw_body) if _raw_body else {}
        except orjson.JSONDecodeError:
            _early_request_body = {}
        _messages_len = len(_early_request_body.get("messages") or [])

        _stats["reqs_total"] += 1
        logger.debug(
            "[session-server] chat_start worker_port=%s session_id=%s req_id=%s messages_len=%d inflight_before=%d",
            worker_port,
            session_id,
            req_id,
            _messages_len,
            _inflight_chat["count"],
        )
        _inflight_chat["count"] += 1
        try:
            session = registry.get_or_create_session(session_id)
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            t_lock_wait_start = time.monotonic()
            async with session.lock:
                t_lock_acquired = time.monotonic()
                # Double-check: session may have been marked closing while waiting for lock.
                if session.closing:
                    raise SessionNotFoundError(f"session not found: session_id={session_id}")

                # Reuse the body/parsed JSON already read up-front for the
                # chat_start log (see top of handler). The body is consumed
                # exactly once by Starlette; reading again here returns empty.
                request_body = _early_request_body

                # TITO token tracking requires three SGLang flags working together:
                #   logprobs=True            → populates meta_info.output_token_logprobs
                #   return_prompt_token_ids  → adds choice.prompt_token_ids
                #   return_meta_info         → wraps the above in choice.meta_info
                # All three are hardcoded (not setdefault) to prevent agent-side
                # overrides from breaking the token accumulation invariants.
                request_body["logprobs"] = True
                request_body["return_prompt_token_ids"] = True
                request_body["return_meta_info"] = True
                request_body["return_sampling_mask"] = session.capture_sampling_mask
                if getattr(args, "use_rollout_routing_replay", False):
                    request_body["return_routed_experts"] = True
                # Must be False so stop tokens are trimmed from output: otherwise the
                # agent sees stop-token text in content, and the accumulated checkpoint
                # would duplicate structural delimiters that the chat template also emits.
                request_body["no_stop_trim"] = False

                request_messages = request_body.get("messages", [])
                # Run the sync tito-tokenizer call in a thread so the event
                # loop isn't blocked while merge_tokens / chat-template render
                # holds the GIL. At 300+ in-flight sessions this shaved ~40%
                # off server p99 in microbench.
                t_tok_in_start = time.monotonic()
                pretokenized = await asyncio.to_thread(
                    session.prepare_pretokenized,
                    request_messages,
                    tools=request_body.get("tools"),
                    tito_tokenizer=registry.tito_tokenizer,
                )
                t_tok_in_end = time.monotonic()
                if pretokenized is not None:
                    request_body["input_ids"] = pretokenized["input_ids"]
                    logger.debug(
                        "Using pretokenized input_ids: %d tokens",
                        len(pretokenized["input_ids"]),
                    )

                body = orjson.dumps(request_body)
                expected_num_assistant = session.num_assistant

            # Tag every turn of this session with a routing key so a routing-key
            # gateway policy (manual / consistent_hashing) pins the session to one
            # worker, reusing the worker that holds its KV cache. Emitted
            # unconditionally: the gateway is launched externally (miles does not
            # know its policy), and policies that don't route on the key
            # (e.g. cache_aware) ignore the header.
            proxy_headers = {**dict(request.headers), "X-SMG-Routing-Key": session_id}

            t_proxy_start = time.monotonic()
            result = await backend.do_proxy(request, "v1/chat/completions", body=body, headers=proxy_headers)
            t_proxy_end = time.monotonic()
            # Session was closed mid-turn: return the engine's response without
            # recording it into the trajectory.
            if session.closing:
                return backend.build_proxy_response(result)

            # If SGLang returned a non-200 error (e.g. 400 for context too long),
            # pass it through to the agent without recording — the agent can retry
            # or handle the error.
            if result["status_code"] != 200:
                # Rollback failures indicate corrupted prefix-cache state in SGLang.
                # Retry once without pretokenized input_ids so SGLang processes the
                # request from scratch instead of attempting prefix continuation.
                error_body = result.get("response_body") or b""
                if isinstance(error_body, bytes):
                    error_body = error_body.decode("utf-8", errors="replace")
                if (
                    result["status_code"] == 400
                    and "rollback failed" in error_body.lower()
                    and "input_ids" in request_body
                ):
                    logger.warning(
                        "SGLang rollback failed for session %s, retrying without prefix continuation",
                        session_id,
                    )
                    request_body.pop("input_ids", None)
                    retry_body = orjson.dumps(request_body)
                    result = await backend.do_proxy(
                        request, "v1/chat/completions", body=retry_body, headers=proxy_headers
                    )
                    t_proxy_end = time.monotonic()
                    if session.closing:
                        return backend.build_proxy_response(result)
                    if result["status_code"] != 200:
                        return backend.build_proxy_response(result)
                else:
                    return backend.build_proxy_response(result)

            response = await asyncio.to_thread(orjson.loads, result["response_body"])

            choice = response.get("choices", [{}])[0]

            meta_info = choice.get("meta_info")
            if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
                raise UpstreamResponseError(
                    "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
                )
            assistant_message = choice.get("message", {})
            if assistant_message.get("content") is None:
                raise UpstreamResponseError(
                    "assistant message content is None, when tool call parser failed SGLang should still return "
                    "an empty content rather than None. Please check your modified SGLang version."
                )

            prompt_token_ids = choice.get("prompt_token_ids")
            output_token_logprobs = meta_info["output_token_logprobs"]
            completion_tokens = meta_info["completion_tokens"]

            actual_output_logprobs_len = len(output_token_logprobs)
            if actual_output_logprobs_len != completion_tokens:
                raise UpstreamResponseError(
                    "invalid chat completion response: "
                    f"len(output_token_logprobs)={actual_output_logprobs_len} "
                    f"!= completion_tokens={completion_tokens}. "
                    f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
                )

            completion_token_ids = [t[1] for t in output_token_logprobs]

            rollout_sampling_mask = None
            rollout_sampling_log_probs = None
            if session.capture_sampling_mask:
                rollout_sampling_mask, rollout_sampling_log_probs = await asyncio.to_thread(
                    _compact_sampling_replay, meta_info
                )
                result["response_body"] = orjson.dumps(response)

            async with session.lock:
                if session.closing:
                    logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                    return backend.build_proxy_response(result)

                if session.num_assistant != expected_num_assistant:
                    # Under the split lock another writer can change num_assistant
                    # between our snapshot and our re-acquire of the lock; log the
                    # mismatch and skip the state update.
                    logger.warning(
                        "[session-server] state_changed_during_proxy "
                        "worker_port=%s session_id=%s req_id=%s "
                        "expected_num_assistant=%d got_num_assistant=%d "
                        "inflight_chat_count=%d caller_request_id=%s "
                        "proxy_elapsed_ms=%.1f",
                        worker_port,
                        session_id,
                        req_id,
                        expected_num_assistant,
                        session.num_assistant,
                        _inflight_chat["count"],
                        request.headers.get("x-request-id", "-"),
                        (t_proxy_end - t_proxy_start) * 1000.0,
                    )
                    return backend.build_proxy_response(result)

                # Same rationale as the prepare_pretokenized call above —
                # offload the sync merge_tokens / state update to a thread
                # so concurrent in-flight sessions keep moving.
                t_tok_out_start = time.monotonic()
                await asyncio.to_thread(
                    session.update_pretokenized_state,
                    request_messages,
                    assistant_message,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
                )
                t_tok_out_end = time.monotonic()

                record = SessionRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path="/v1/chat/completions",
                    status_code=result["status_code"],
                    request=request_body,
                    response=response,
                    rollout_sampling_mask=rollout_sampling_mask,
                    rollout_sampling_log_probs=rollout_sampling_log_probs,
                )
                session.append_record(record)

                try:
                    prompt_tokens_emit = int(meta_info.get("prompt_tokens", -1))
                except (TypeError, ValueError):
                    prompt_tokens_emit = -1
                try:
                    completion_tokens_emit = int(meta_info.get("completion_tokens", -1))
                except (TypeError, ValueError):
                    completion_tokens_emit = -1
                _stats["turns_completed"] += 1

            return backend.build_proxy_response(result)
        finally:
            _inflight_chat["count"] -= 1
            t_handler_end = time.monotonic()
            # One DEBUG log per request, irrespective of which path the handler
            # took (success, 404, upstream error). Spans that didn't run come
            # through as 0.0 ms — a valid signal ("we never got there").
            logger.debug(
                "[session-server] chat_done worker_port=%s session_id=%s req_id=%s lock_wait_ms=%.1f tokenize_in_ms=%.1f proxy_elapsed_ms=%.1f tokenize_out_ms=%.1f total_ms=%.1f inflight_now=%d prompt_tokens=%d completion_tokens=%d",
                worker_port,
                session_id,
                req_id,
                (t_lock_acquired - t_lock_wait_start) * 1000.0,
                (t_tok_in_end - t_tok_in_start) * 1000.0,
                (t_proxy_end - t_proxy_start) * 1000.0,
                (t_tok_out_end - t_tok_out_start) * 1000.0,
                (t_handler_end - t_handler_start) * 1000.0,
                _inflight_chat["count"],
                prompt_tokens_emit,
                completion_tokens_emit,
            )

    @app.get("/sessions/{session_id}/v1/model_info")
    async def session_v1_model_info(session_id: str):
        """OpenAI-compatible /v1/model_info stub.

        LiteLLM (and other OpenAI-compat clients) probe /v1/model_info on
        first connect for tokenizer hash and model-ID discovery. SGLang
        does not expose /v1/model_info (it exposes /model_info without
        the /v1/ prefix), so the default wildcard proxy below returns
        404 and clients emit a warning on every new session.

        Return a minimal stub advertising the served model name. Does
        not validate the session_id because /v1/model_info is
        session-independent; the session path segment is retained to
        keep URL shape consistent with the other session endpoints.
        """
        return {
            "id": getattr(args, "sglang_served_model_name", None) or "model",
            "object": "model",
        }

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
