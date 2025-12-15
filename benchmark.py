#!/usr/bin/env python3
"""
OpenAI Responses API concurrency / throughput stress test.

Measures:
- Job completion rate (unique requests / sec)
- True API call throughput (attempts / sec, includes retries)
- Latency percentiles for successful requests (P50/P90/P99)
"""

import argparse
import asyncio
import json
import logging
import os
import random
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import aiohttp

RESPONSES_API_URL = "https://api.openai.com/v1/responses"


@dataclass
class RequestResult:
    ok: bool
    attempts: int
    latency_ms: float | None  # Final successful attempt only
    total_duration_ms: float  # Wall time including retries
    status: int | None
    response_id: str | None
    error: Any | None


DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_VERBOSITY = "low"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 16000


def load_template_first_line(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
    if not line:
        raise ValueError(f"Template file is empty: {path}")
    return json.loads(line)


def make_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def parse_retry_after(header_value: str | None, max_retry_after_s: float = 5.0) -> float | None:
    if not header_value:
        return None
    try:
        seconds = float(header_value)
        if seconds < 0:
            return None
        return min(seconds, max_retry_after_s)
    except ValueError:
        return None


def percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def build_responses_payload(template: dict[str, Any], request_index: int) -> dict[str, Any]:
    """
    Builds a payload for the OpenAI Responses API, injecting a unique UUID to prevent prompt caching.

    Template should have an 'input' field (string or list of messages).
    """
    unique_id = uuid.uuid4().hex
    input_data = template.get("input", "")

    # Handle both string and array input formats
    if isinstance(input_data, str):
        input_with_id = f"[stress_test_id={unique_id} request_index={request_index}]\n{input_data}"
    elif isinstance(input_data, list):
        # Inject UUID into the first user message
        input_with_id = []
        uuid_injected = False
        for msg in input_data:
            if isinstance(msg, dict):
                new_msg = dict(msg)
                if not uuid_injected and msg.get("role") == "user":
                    content = msg.get("content", "")
                    new_msg["content"] = f"[stress_test_id={unique_id} request_index={request_index}]\n{content}"
                    uuid_injected = True
                input_with_id.append(new_msg)
            else:
                input_with_id.append(msg)
    else:
        input_with_id = input_data

    payload = {
        "model": DEFAULT_MODEL,
        "input": input_with_id,
        "text": {"verbosity": DEFAULT_VERBOSITY},
        "reasoning": {"effort": DEFAULT_REASONING_EFFORT},
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }

    return payload


def is_retryable_http(status: int) -> bool:
    return status == 429 or status == 408 or 500 <= status <= 599


async def post_once(
    session: aiohttp.ClientSession,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[int, Any, float | None]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(
        url=RESPONSES_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
        raise_for_status=False,
    ) as resp:
        status = resp.status
        retry_after = parse_retry_after(resp.headers.get("Retry-After"))
        try:
            body: Any = await resp.json()
        except Exception:
            body = await resp.text()
        return status, body, retry_after


async def run_one_request(
    session: aiohttp.ClientSession,
    headers: dict[str, str],
    template: dict[str, Any],
    request_index: int,
    timeout_s: float,
    retries: int,
    retry_delay_s: float,
    jitter_s: float = 0.1,
) -> RequestResult:
    payload = build_responses_payload(template, request_index)
    overall_start = time.perf_counter()

    attempts = 0
    last_status: int | None = None
    last_error: Any | None = None
    last_latency_ms: float | None = None
    server_retry_after: float | None = None

    for attempt in range(retries + 1):
        attempts += 1
        start = time.perf_counter()
        logging.debug(f"Req {request_index} attempt {attempts}: sending request to {RESPONSES_API_URL}")

        try:
            status, body, retry_after = await post_once(
                session=session,
                headers=headers,
                payload=payload,
                timeout_s=timeout_s,
            )
            last_latency_ms = (time.perf_counter() - start) * 1000.0

            if 200 <= status < 300:
                total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
                response_id = body.get("id") if isinstance(body, dict) else None
                return RequestResult(
                    ok=True,
                    attempts=attempts,
                    latency_ms=last_latency_ms,
                    total_duration_ms=total_duration_ms,
                    status=status,
                    response_id=response_id,
                    error=None,
                )

            last_status = status
            last_error = body

            if status == 429:
                server_retry_after = retry_after
                logging.debug(f"Req {request_index} attempt {attempts}: HTTP 429 (rate limited)")
            else:
                logging.debug(f"Req {request_index} attempt {attempts}: HTTP {status}")

            retryable = is_retryable_http(status)

        except asyncio.TimeoutError:
            last_status = None
            last_error = f"timeout after {timeout_s}s"
            retryable = True
            logging.debug(f"Req {request_index} attempt {attempts}: timeout after {timeout_s}s")

        except aiohttp.ClientError as e:
            last_status = None
            last_error = f"aiohttp client error: {e}"
            retryable = True
            logging.debug(f"Req {request_index} attempt {attempts}: network error: {e}")

        except Exception as e:
            last_status = None
            last_error = f"unexpected error: {e}"
            retryable = True
            logging.debug(f"Req {request_index} attempt {attempts}: unexpected error: {e}")

        if attempt >= retries or not retryable:
            break

        # Backoff before next attempt
        if server_retry_after is not None:
            backoff = server_retry_after + random.uniform(0, jitter_s)
            server_retry_after = None
        elif retry_delay_s > 0:
            backoff = retry_delay_s * (2 ** attempt) + random.uniform(0, jitter_s)
        else:
            backoff = random.uniform(0, jitter_s)

        if backoff > 0:
            await asyncio.sleep(backoff)

    total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
    return RequestResult(
        ok=False,
        attempts=attempts,
        latency_ms=last_latency_ms,
        total_duration_ms=total_duration_ms,
        status=last_status,
        response_id=None,
        error=last_error,
    )


async def run_stress_test(
    *,
    api_key: str,
    template: dict[str, Any],
    num_requests: int,
    concurrency: int,
    timeout_s: float,
    retries: int,
    retry_delay_s: float,
    ramp_up_s: float,
    out_path: str | None,
    progress_every: int,
    baseline_duration_ms: float | None,
) -> None:
    headers = make_headers(api_key)

    connector = aiohttp.TCPConnector(limit=concurrency)

    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(num_requests):
        queue.put_nowait(i)

    # Shared aggregates (single-threaded under asyncio, but still keep operations simple)
    completed = 0
    succeeded = 0
    total_attempts = 0
    success_latencies: list[float] = []
    success_total_durations: list[float] = []
    start_wall = time.time()

    lock = asyncio.Lock()  # keep writes / shared counters consistent

    with (open(out_path, "w", encoding="utf-8") if out_path else nullcontext()) as out_fh:

        async def worker(worker_id: int, session: aiohttp.ClientSession) -> None:
            nonlocal completed, succeeded, total_attempts

            while True:
                try:
                    request_index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                result = await run_one_request(
                    session=session,
                    headers=headers,
                    template=template,
                    request_index=request_index,
                    timeout_s=timeout_s,
                    retries=retries,
                    retry_delay_s=retry_delay_s,
                )
                logging.debug(f"Request {request_index} completed in {result.total_duration_ms:.0f}ms (latency={result.latency_ms or 0:.0f}ms, ok={result.ok})")

                async with lock:
                    completed += 1
                    total_attempts += result.attempts
                    if result.ok:
                        succeeded += 1
                        if result.latency_ms is not None:
                            success_latencies.append(result.latency_ms)
                        success_total_durations.append(result.total_duration_ms)

                    if out_fh:
                        row = {
                            "request_index": request_index,
                            "ok": result.ok,
                            "attempts": result.attempts,
                            "latency_ms": result.latency_ms,
                            "total_duration_ms": result.total_duration_ms,
                            "status": result.status,
                            "response_id": result.response_id,
                        }
                        if not result.ok:
                            row["error"] = result.error
                        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

                    if progress_every > 0 and completed % progress_every == 0:
                        elapsed = time.time() - start_wall
                        job_rps = completed / elapsed if elapsed > 0 else 0.0
                        api_rps = total_attempts / elapsed if elapsed > 0 else 0.0
                        logging.info(
                            f"Progress: {completed}/{num_requests} done | "
                            f"ok={succeeded} | job_rps={job_rps:.2f} | api_rps={api_rps:.2f}"
                        )

        def ramp_delay_for_worker(worker_id: int) -> float:
            if ramp_up_s <= 0 or concurrency <= 1:
                return 0.0
            step = ramp_up_s / max(concurrency - 1, 1)
            return worker_id * step

        logging.info(f"Starting stress test with {concurrency} workers, {num_requests} requests")
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks: list[asyncio.Task[None]] = []
            for wid in range(concurrency):
                delay = ramp_delay_for_worker(wid)
                if delay > 0:
                    # Stagger worker start to avoid burst at t=0
                    async def delayed_start(w: int, d: float, s: aiohttp.ClientSession) -> None:
                        await asyncio.sleep(d)
                        await worker(w, s)

                    tasks.append(asyncio.create_task(delayed_start(wid, delay, session)))
                else:
                    tasks.append(asyncio.create_task(worker(wid, session)))

            await asyncio.gather(*tasks)

    elapsed = time.time() - start_wall

    # Summary
    failed = completed - succeeded
    job_rps = completed / elapsed if elapsed > 0 else 0.0
    api_rps = total_attempts / elapsed if elapsed > 0 else 0.0

    success_latencies.sort()
    p50 = percentile(success_latencies, 50)
    p90 = percentile(success_latencies, 90)
    p95 = percentile(success_latencies, 95)
    p99 = percentile(success_latencies, 99)
    max_latency = success_latencies[-1] if success_latencies else 0.0
    avg = (sum(success_latencies) / len(success_latencies)) if success_latencies else 0.0

    success_total_durations.sort()
    avg_total = (sum(success_total_durations) / len(success_total_durations)) if success_total_durations else 0.0
    p50_total = percentile(success_total_durations, 50)
    p90_total = percentile(success_total_durations, 90)
    p95_total = percentile(success_total_durations, 95)
    p99_total = percentile(success_total_durations, 99)
    max_total = success_total_durations[-1] if success_total_durations else 0.0

    logging.info("============================================================")
    logging.info("THROUGHPUT TEST COMPLETE")
    logging.info("============================================================")
    if out_path:
        logging.info(f"Results file:         {out_path}")
    logging.info(f"Total requests:       {completed}")
    logging.info(f"Total API calls:      {total_attempts}")
    logging.info(f"Final succeeded:      {succeeded}")
    logging.info(f"Final failed:         {failed}")
    logging.info("--- Latency (final successful attempt) ---")
    logging.info(f"Avg latency:          {avg:.1f}ms")
    logging.info(f"P50 latency:          {p50:.1f}ms")
    logging.info(f"P90 latency:          {p90:.1f}ms")
    logging.info(f"P95 latency:          {p95:.1f}ms")
    logging.info(f"P99 latency:          {p99:.1f}ms")
    logging.info(f"Max latency:          {max_latency:.1f}ms")
    logging.info("--- Total Duration (including retries) ---")
    logging.info(f"Avg total duration:   {avg_total:.1f}ms")
    logging.info(f"P50 total duration:   {p50_total:.1f}ms")
    logging.info(f"P90 total duration:   {p90_total:.1f}ms")
    logging.info(f"P95 total duration:   {p95_total:.1f}ms")
    logging.info(f"P99 total duration:   {p99_total:.1f}ms")
    logging.info(f"Max total duration:   {max_total:.1f}ms")
    logging.info("-----------------------------------------------------------")
    logging.info(f"Concurrency:          {concurrency}")
    logging.info(f"Ramp-up:              {ramp_up_s}s")
    logging.info(f"Timeout:              {timeout_s}s")
    logging.info(f"Retries:              {retries}")
    logging.info(f"Retry delay:          {retry_delay_s}s")
    logging.info(f"Elapsed:              {elapsed * 1000:.0f}ms ({elapsed:.2f}s)")
    logging.info("--- Throughput ---")
    logging.info(f"Job completion rate:  {job_rps:.2f} req/s ({job_rps * 60:.0f} RPM)")
    logging.info(f"API call throughput:  {api_rps:.2f} req/s ({api_rps * 60:.0f} RPM)")
    if baseline_duration_ms is not None:
        baseline_duration_s = baseline_duration_ms / 1000.0
        theoretical_rps = concurrency / baseline_duration_s
        efficiency = (job_rps / theoretical_rps) * 100.0 if theoretical_rps > 0 else 0.0
        logging.info(f"Theoretical RPS:      {theoretical_rps:.2f} req/s (concurrency={concurrency}, baseline={baseline_duration_ms:.0f}ms)")
        logging.info(f"Efficiency:           {efficiency:.1f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI Responses API throughput stress test")
    p.add_argument("--requests-file", required=True, help="JSONL with a single template request on line 1")
    p.add_argument("--num-requests", type=int, required=True, help="How many requests to send")
    p.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--concurrency", type=int, default=50, help="Max concurrent in-flight requests")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-attempt timeout in seconds")
    p.add_argument("--retries", type=int, default=0, help="Retries per request (0 = no retries)")
    p.add_argument("--retry-delay", type=float, default=0.0, help="Base delay in seconds (exponential backoff)")
    p.add_argument("--ramp-up", type=float, default=0.0, help="Ramp-up window (seconds) to stagger worker starts")
    p.add_argument("--out", default=None, help="Write per-request results to JSONL (optional)")
    p.add_argument("--progress-every", type=int, default=0, help="Log progress every N completed requests (0 = off)")
    p.add_argument("--baseline-duration-ms", type=float, default=None, help="Total duration (ms) of a single request at concurrency=1, including retries. Used to compute efficiency.")
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = p.parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    if args.retry_delay < 0:
        raise ValueError("--retry-delay must be >= 0")
    if args.ramp_up < 0:
        raise ValueError("--ramp-up must be >= 0")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")
    if args.baseline_duration_ms is not None and args.baseline_duration_ms <= 0:
        raise ValueError("--baseline-duration-ms must be > 0")

    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    template = load_template_first_line(args.requests_file)
    logging.info(f"Input document length: {len(json.dumps(template)):,} characters")

    asyncio.run(
        run_stress_test(
            api_key=api_key,
            template=template,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
            retries=args.retries,
            retry_delay_s=args.retry_delay,
            ramp_up_s=args.ramp_up,
            out_path=args.out,
            progress_every=args.progress_every,
            baseline_duration_ms=args.baseline_duration_ms,
        )
    )


if __name__ == "__main__":
    main()
