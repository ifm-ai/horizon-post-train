"""Fake SGLang worker — HTTP stub for gateway and Miles registration testing.

Implements just enough of the SGLang http_server.py surface for:
  1. sgl-model-gateway metadata discovery (/server_info, /model_info)
  2. Gateway health probes (/health, /health_generate)
  3. Request routing through /v1/chat/completions with optional TITO
     extras (completion_token_ids, routed_experts) when the caller sets
     extra_body.return_completion_token_ids / return_routed_experts.
  4. Miles weight-sync RPCs (/init_weights_update_group,
     /update_weights_from_distributed, /flush_cache, /abort_request) so
     the RolloutManager actor doesn't crash during startup.

Does NOT run a real model. All completions are canned text + synthetic
integer token IDs. Use to shake out registration, routing, and weight-sync
paths end-to-end without launching any GPU engine.

Usage:
    python3 fake_sglang_worker.py --host 0.0.0.0 --port 30000 \\
        --worker-type decode --served-model-name glm-4.7-flash

Two-worker PD disagg smoke-test example:
    # Terminal 1 -- prefill with bootstrap port
    python3 fake_sglang_worker.py --port 30000 --worker-type prefill \\
        --bootstrap-port 8998 --disaggregation-mode prefill

    # Terminal 2 -- decode
    python3 fake_sglang_worker.py --port 30001 --worker-type decode \\
        --disaggregation-mode decode

    # Terminal 3 -- gateway + register
    sgl-model-gateway --port 8080 --policy cache_aware &
    curl -X POST 'http://localhost:8080/workers' \\
        -H 'content-type: application/json' \\
        -d '{"url": "http://localhost:30000", "worker_type": "prefill", \\
             "bootstrap_port": 8998}'
    curl -X POST 'http://localhost:8080/workers' \\
        -H 'content-type: application/json' \\
        -d '{"url": "http://localhost:30001", "worker_type": "decode"}'
    curl 'http://localhost:8080/list_workers'
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

logger = logging.getLogger("fake_sglang_worker")


def build_app(config: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="fake-sglang-worker")

    # Deterministic tokenizer sha so TITO consistency checks against a real
    # worker can be simulated (agent sends tokenizer_sha256 in request metadata
    # and the trainer compares). When two fake workers use the same
    # --served-model-name, the sha matches.
    tok_sha = hashlib.sha256(config.served_model_name.encode("utf-8")).hexdigest()

    @app.get("/health")
    @app.get("/health_generate")
    async def health() -> Response:
        return Response(status_code=200)

    @app.get("/get_model_info")
    @app.get("/model_info")
    async def model_info() -> dict:
        return {
            "model_path": config.model_path,
            "tokenizer_path": config.tokenizer_path,
            "tokenizer_sha256": tok_sha,
            "is_generation": True,
            "model_type": config.model_type,
            "architectures": [config.architecture],
            "weight_version": config.weight_version,
            "preferred_sampling_params": None,
            "has_image_understanding": False,
            "has_audio_understanding": False,
        }

    @app.get("/get_server_info")
    @app.get("/server_info")
    async def server_info() -> dict:
        return {
            "model": config.served_model_name,
            "model_id": config.served_model_name,
            "model_path": config.model_path,
            "served_model_name": config.served_model_name,
            "tp_size": config.tp_size,
            "dp_size": config.dp_size,
            "load_balance_method": "round_robin",
            "disaggregation_mode": config.disaggregation_mode,
            "version": "fake-0.0.1",
            "max_batch_size": 256,
            "max_total_tokens": 32768,
            "max_prefill_tokens": 16384,
            "max_running_requests": 64,
            "max_num_reqs": 64,
        }

    @app.get("/get_load")
    async def get_load() -> dict:
        # Gateway load-probe polls this. Return something plausible so the
        # CacheAwarePolicy has numbers to route on.
        return {
            "waiting": 0,
            "running": 0,
            "gen_throughput": 0.0,
        }

    @app.get("/hicache/storage-backend")
    async def hicache_backend() -> dict:
        # Useful for HiCache L3 smoke-testing: report a canned mooncake
        # backend presence so test scripts can assert end-to-end.
        return {
            "backend": config.hicache_backend or "none",
            "connected": bool(config.hicache_backend),
        }

    @app.get("/parallelism_config")
    async def parallelism_config() -> dict:
        return {"tp_size": config.tp_size, "dp_size": config.dp_size, "pp_size": 1}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request) -> JSONResponse:
        body = await req.json()
        # SGLang accepts return_completion_token_ids / return_routed_experts
        # as TOP-LEVEL fields on ChatCompletionRequest. The OpenAI client's
        # `extra_body={...}` convention flattens those dict entries into the
        # top-level JSON body. Some callers still nest them — accept both.
        extra = body.get("extra_body") or {}
        return_cti = bool(
            body.get("return_completion_token_ids")
            or extra.get("return_completion_token_ids")
        )
        return_rex = bool(
            body.get("return_routed_experts")
            or extra.get("return_routed_experts")
        )

        t0 = time.time()
        # Simulate a tiny amount of work proportional to prompt length so
        # throughput experiments produce non-trivial numbers.
        prompt_len = sum(len(m.get("content", "")) for m in body.get("messages", []))
        time_to_simulate = min(0.05 + prompt_len / 50_000.0, 0.5)
        # Intentionally NOT sleeping here — we want a worker this fast to be
        # the baseline for throughput tests (the gateway + wire is the
        # interesting bit). Flip FAKE_WORKER_SLEEP=1 if you want latency.
        _ = time_to_simulate  # silence lint

        response_text = "Fake response from debug worker."
        # Synthetic IDs — don't need real vocab IDs for the gateway path;
        # Miles will decode them with the real tokenizer when it runs
        # compute_samples_from_openai_records, but mock_agent_function.py
        # is the canonical consumer and it just reads reward.
        completion_token_ids = [random.randint(100, 150_000) for _ in range(32)]

        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }
        if return_cti:
            choice["completion_token_ids"] = completion_token_ids
        if return_rex:
            # Shape: list[list[int]] per-layer per-token — keep tiny.
            choice["routed_experts"] = [[0, 1] for _ in completion_token_ids[:4]]

        resp = {
            "id": f"chatcmpl-fake-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(t0),
            "model": body.get("model", config.served_model_name),
            "choices": [choice],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": len(completion_token_ids),
                "total_tokens": 8 + len(completion_token_ids),
            },
        }
        return JSONResponse(resp)

    @app.post("/generate")
    async def generate(req: Request) -> JSONResponse:
        body = await req.json()
        sp = body.get("sampling_params") or {}
        n = int(sp.get("max_new_tokens", 16))
        return JSONResponse(
            {
                "text": "fake",
                "output_ids": [random.randint(100, 150_000) for _ in range(n)],
                "meta_info": {
                    "completion_tokens": n,
                    "prompt_tokens": 8,
                    "finish_reason": {"type": "stop"},
                },
            }
        )

    # Miles weight-sync RPCs — all no-op 200s.
    @app.post("/init_weights_update_group")
    @app.post("/update_weights_from_distributed")
    @app.post("/update_weights_from_tensor")
    @app.post("/destroy_weights_update_group")
    @app.post("/flush_cache")
    @app.post("/abort_request")
    @app.post("/pause_generation")
    @app.post("/continue_generation")
    @app.post("/post_process_weights")
    @app.post("/update_weight_version")
    async def weight_noop(req: Request) -> JSONResponse:
        # Return the same shape SGLang returns so Miles's response parsers
        # don't choke.
        return JSONResponse({"status": "ok"})

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fake SGLang worker for gateway testing.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument(
        "--worker-type",
        default="regular",
        choices=["regular", "prefill", "decode"],
        help="Reported to gateway for PD routing. 'regular' means non-disagg.",
    )
    p.add_argument(
        "--bootstrap-port",
        type=int,
        default=None,
        help="Advertised in /server_info when --worker-type=prefill.",
    )
    p.add_argument(
        "--disaggregation-mode",
        default=None,
        choices=[None, "null", "prefill", "decode"],
        help="Server-reported PD mode. Falls back to --worker-type if unset.",
    )
    p.add_argument("--served-model-name", default="fake-glm-4.7-flash")
    p.add_argument("--model-path", default="/fake/path/glm-4.7-flash")
    p.add_argument("--tokenizer-path", default="/fake/path/glm-4.7-flash")
    p.add_argument("--model-type", default="glm4")
    p.add_argument("--architecture", default="Glm4ForCausalLM")
    p.add_argument("--weight-version", default="fake-v0")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument(
        "--hicache-backend",
        default=None,
        help="Reported in /hicache/storage-backend, e.g. 'mooncake' for L3 test.",
    )
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    # Infer disaggregation_mode from worker_type if not explicitly set, so
    # /server_info matches what the gateway expects for PD routing.
    if args.disaggregation_mode is None:
        args.disaggregation_mode = {
            "regular": "null",
            "prefill": "prefill",
            "decode": "decode",
        }[args.worker_type]
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s fake-worker] %(message)s"
    )
    args = parse_args()
    app = build_app(args)
    logger.info(
        "fake SGLang worker starting on %s:%d as worker_type=%s (disagg=%s, "
        "bootstrap_port=%s, model=%s)",
        args.host,
        args.port,
        args.worker_type,
        args.disaggregation_mode,
        args.bootstrap_port,
        args.served_model_name,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
