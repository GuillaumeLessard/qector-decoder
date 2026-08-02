# QECTOR MCP Benchmark Report

Generated (UTC): 2026-08-02T05:59:13.294039+00:00
Server: `{'name': 'qector-decoder-v3', 'version': '0.7.0'}`
Host: Linux-6.6.122+-x86_64-with-glibc2.35 | Python 3.12.13

## Tools

- `decode_syndrome`
- `batch_decode`
- `decode_hyperedge`
- `decode_syndrome_blossom`
- `batch_decode_blossom`
- `decode_syndrome_cascade`
- `benchmark_decoder`
- `run_ler_benchmark`
- `get_decoder_info`
- `get_backend_health`
- `clear_decoder_cache`
- `get_server_env`
- `recommend_decoder`

## Benchmarks (repetition / ring)

| code | n | decoder | samples | throughput | mean_us | unfaithful |
|------|---|---------|---------|------------|---------|------------|
| repetition | 5 | unionfind | 8000 | 11462979 | 0.087 | 0 |
| repetition | 5 | fastunionfind | 8000 | 11540387 | 0.087 | 0 |
| repetition | 5 | blossom | 8000 | 8262646 | 0.121 | 0 |
| repetition | 5 | sparseblossom | 8000 | 5236849 | 0.191 | 0 |
| repetition | 5 | bposd | 8000 | 314525 | 3.179 | 0 |
| repetition | 5 | auto | 8000 | 8261043 | 0.121 | 0 |
| repetition | 9 | unionfind | 4000 | 9877914 | 0.101 | 0 |
| repetition | 9 | fastunionfind | 4000 | 10000813 | 0.1 | 0 |
| repetition | 9 | blossom | 4000 | 5070210 | 0.197 | 0 |
| repetition | 9 | sparseblossom | 4000 | 1980725 | 0.505 | 0 |
| repetition | 9 | bposd | 4000 | 94640 | 10.566 | 0 |
| repetition | 9 | auto | 4000 | 3177478 | 0.315 | 0 |
| repetition | 17 | unionfind | 2500 | 4242251 | 0.236 | 0 |
| repetition | 17 | fastunionfind | 2500 | 4472615 | 0.224 | 0 |
| repetition | 17 | blossom | 2500 | 1695841 | 0.59 | 0 |
| repetition | 17 | sparseblossom | 2500 | 1136156 | 0.88 | 0 |
| repetition | 17 | bposd | 2500 | 79299 | 12.61 | 0 |
| repetition | 17 | auto | 2500 | 3357187 | 0.298 | 0 |
| repetition | 25 | unionfind | 1500 | 6178188 | 0.162 | 0 |
| repetition | 25 | fastunionfind | 1500 | 6300252 | 0.159 | 0 |
| repetition | 25 | blossom | 1500 | 2159946 | 0.463 | 0 |
| repetition | 25 | sparseblossom | 1500 | 1521550 | 0.657 | 0 |
| repetition | 25 | bposd | 1500 | 52477 | 19.056 | 0 |
| repetition | 25 | auto | 1500 | 2151542 | 0.465 | 0 |
| repetition | 33 | unionfind | 1000 | 4932787 | 0.203 | 0 |
| repetition | 33 | fastunionfind | 1000 | 4953127 | 0.202 | 0 |
| repetition | 33 | blossom | 1000 | 1536840 | 0.651 | 0 |
| repetition | 33 | sparseblossom | 1000 | 1165277 | 0.858 | 0 |
| repetition | 33 | bposd | 1000 | 45271 | 22.089 | 0 |
| repetition | 33 | auto | 1000 | 1292897 | 0.773 | 0 |
| repetition | 49 | unionfind | 600 | 3346108 | 0.299 | 0 |
| repetition | 49 | fastunionfind | 600 | 3512604 | 0.285 | 0 |
| repetition | 49 | blossom | 600 | 1010423 | 0.99 | 0 |
| repetition | 49 | sparseblossom | 600 | 774424 | 1.291 | 0 |
| repetition | 49 | bposd | 600 | 27953 | 35.773 | 0 |
| repetition | 49 | auto | 600 | 985420 | 1.015 | 0 |
| repetition | 65 | unionfind | 400 | 2900839 | 0.345 | 0 |
| repetition | 65 | fastunionfind | 400 | 2754975 | 0.363 | 0 |
| repetition | 65 | blossom | 400 | 172643 | 5.792 | 0 |
| repetition | 65 | sparseblossom | 400 | 253455 | 3.945 | 0 |
| repetition | 65 | bposd | 400 | 19176 | 52.148 | 0 |
| repetition | 65 | auto | 400 | 251774 | 3.972 | 0 |
| ring | 16 | unionfind | 3000 | 6791172 | 0.147 | 0 |
| ring | 16 | fastunionfind | 3000 | 7687193 | 0.13 | 0 |
| ring | 16 | blossom | 3000 | 3940735 | 0.254 | 0 |
| ring | 16 | auto | 3000 | 4104082 | 0.244 | 0 |
| ring | 32 | unionfind | 1500 | 5100896 | 0.196 | 0 |
| ring | 32 | fastunionfind | 1500 | 5183528 | 0.193 | 0 |
| ring | 32 | blossom | 1500 | 1877142 | 0.533 | 0 |
| ring | 32 | auto | 1500 | 1883361 | 0.531 | 0 |
| ring | 48 | unionfind | 800 | 3377505 | 0.296 | 0 |
| ring | 48 | fastunionfind | 800 | 3448535 | 0.29 | 0 |
| ring | 48 | blossom | 800 | 1064594 | 0.939 | 0 |
| ring | 48 | auto | 800 | 1038234 | 0.963 | 0 |

## Summary

```json
{
  "package": "qector-decoder-v3",
  "version": "0.7.0",
  "transport": "MCP stdio JSON-RPC 2.0",
  "timestamp_utc": "2026-08-02T05:59:13.294039+00:00",
  "tools_count": 13,
  "correctness_cases": 42,
  "correctness_faithful_all": true,
  "benchmark_points": 54,
  "zero_unfaithful_in_benchmarks": true,
  "peak_throughput_shots_per_s": 11540387
}
```
