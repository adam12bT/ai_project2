# Optimization protocol

The baseline uses hybrid retrieval and reranks 24 candidates; optimized reranks 8. Within each benchmark, both use the same generator, index, models, output budget and questions. The offline benchmark uses extraction; the Groq benchmark uses real generation. Response caches are disabled. Each question is evaluated under both profiles after local-model warm-up. Test IDs are even, so the current test run evaluates optimized before baseline on each question; order effects are a limitation. Timings include provider backoff when encountered, but exclude the explicit pacing sleep between requests.

The target is >=50% lower end-to-end p95 with <=2 percentage points lost on the correctness proxy. The script reports `target_met` from actual timings and proxy accuracy. This proxy must be replaced/confirmed by human correctness judgments before claiming the assignment's accuracy target. The optimization is not assumed successful in advance. Fewer candidates can lose answers that originally ranked 9–24. A cache is also available for normal requests, but cached repeats do not count toward the primary speedup claim.

Read `results/extractive/metrics_test.json` and `results/groq/metrics_test.json` for separate measurements. Saved report tables reflect the completed measurements; report-generation helpers have been removed. If the target fails, report the failure and choose any further strategy using development data. Never rewrite measurements to meet the target.

## Cost model

Retrieval runs locally and generation uses Groq. With GROQ_FREE_TIER=true, API cost is recorded as zero under the account allowance, subject to quotas. A 50% billing improvement from zero is meaningless, so the main comparison targets latency. For paid-account estimates, use actual prompt/completion tokens and configured current prices. Offline and Groq results are separate because generation/network latency can dominate local reranking savings.

For a host costing H dollars/month and N requests/month, allocated hosting cost/request = H/N before variable expenses. At 100x traffic, fixed hosting would be H/(100N) only if that hardware still has sufficient capacity. Estimate serialized service utilization as arrival_rate * mean_service_seconds for this serialized service. At sustained utilization near 1, queues and p95 will increase; scale and remeasure rather than extrapolate fixed latency. Electricity/request = watts/1000 * service_seconds/3600 * price_per_kWh, plus idle allocation. Capture actual host pricing separately; no cloud prices have been invented.

## Real-user evidence

Feedback is pending. The app records real votes in `data/feedback.jsonl`; those records are ignored by git. Have classmates try their own Python questions, then summarize the count, helpfulness rate, failed questions and resulting changes. Do not submit names, student IDs or private code.

<!-- measured -->
## Actual benchmark

| Profile | Correctness proxy | p50 ms | p95 ms |
|---|---:|---:|---:|
| optimized | 0.877 | 1411.2 | 5451.8 |
| baseline | 0.846 | 1626.6 | 6930.0 |

Measured p95 reduction: 21.3%. Accuracy proxy loss: -3.08 points. Numeric target met on proxy: False. Human correctness validation remains pending.
