# OpenAI Responses API Benchmark

Concurrency and throughput stress test for the OpenAI Responses API.

## Requirements

- Python 3.10+
- OpenAI API key

## Setup

There's only one dependency aiohttp:

```bash
pip install -r requirements.txt
```

## Usage

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Run the benchmark:

```bash
python benchmark.py \
    --requests-file requests.jsonl \
    --num-requests 100 \
    --concurrency 10
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--requests-file` | JSONL file with a template request on line 1 (only `input` field is used) |
| `--num-requests` | Total number of requests to send |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--concurrency` | 50 | Max concurrent in-flight requests |
| `--timeout` | 30.0 | Per-attempt timeout in seconds |
| `--retries` | 0 | Retries per request (0 = no retries) |
| `--retry-delay` | 0.0 | Base delay in seconds (exponential backoff) |
| `--ramp-up` | 0.0 | Seconds to stagger worker starts |
| `--out` | None | Write per-request results to JSONL |
| `--progress-every` | 0 | Log progress every N completed requests |
| `--baseline-duration-ms` | None | Expected single-request duration (for efficiency calculation) |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |
