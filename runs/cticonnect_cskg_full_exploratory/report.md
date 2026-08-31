# PX-034 full CTIConnect CSKG exploratory run

> **Superseded.** This anchor-only result is retained for provenance. Use the [question-conditioned v2 run](../cticonnect_cskg_question_conditioned_exploratory_v2/report.md) and consult the repository [CHANGELOG](../../CHANGELOG.md) for corrected numbers.

Status: **execution pass; exploratory retrieval result only**

This run executed the shipped CTIConnect CSKG entity index over all 341 multi-document synthesis queries at top-k 10. Because CTIConnect v1.0.0 does not expose an independent anchor field, the run used the first member of each gold `blog_ids` cluster as the anchor. This policy is recorded as `exploratory_first_gold_cluster_source` and is not valid for confirmatory claims.

## Execution integrity

- Queries attempted/completed: 341/341
- Unique query IDs: 341
- Detector inputs produced: 341
- Runtime errors: 0
- Forbidden answer/gold fields in detector inputs: 0
- Regression tests after execution: 14/14 passed
- CTIConnect commit: `554797d69a51147f1f98fad7198cb2d2b183d0e9`
- CTIConnect manifest SHA-256: `1c9e3577137333201640b8890221cfee11a64e0eedef4d4def9152c32ad19963`

## Exploratory retrieval results

| Task | Queries | Mean recall@10 | Full-cluster recovery | MRR | Mean precision@10 | Median latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| CSC | 111 | 0.8919 | 0.8198 | 0.9640 | 0.1757 | 2.41 ms | 4.04 ms |
| MLA | 95 | 0.7211 | 0.5684 | 0.6053 | 0.1400 | 2.26 ms | 4.98 ms |
| TAP | 135 | 0.8148 | 0.6963 | 0.7473 | 0.1630 | 2.23 ms | 3.39 ms |
| **Overall** | **341** | **0.8138** | **0.7009** | **0.7783** | **0.1607** | — | — |

Recall treats every non-anchor member of the released gold report cluster as relevant. Precision uses the ten retrieved reports as denominator. Latency covers local BM25 retrieval and instrumentation, not offline CSKG construction.

## Interpretation boundary

This validates full local execution of the CSKG retrieval and option-blind evidence pipeline. It does not test the PX-034 support-state classifier, routing AURC, retrieval-induced harm, oracle-gap recovery, cross-model effects, or annotation taxonomy. Those require independently frozen anchors, expert support-state annotations, dense vanilla/EtR/DtR traces, answer-generation arms, and model/evaluator runs.
