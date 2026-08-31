# PX-034 question-conditioned CTIConnect CSKG exploratory run (v2)

Status: **execution pass; exploratory retrieval result only**

This run uses question-conditioned CSKG retrieval: normalized BM25 scores from question tokens (0.35) are fused with anchor-vocabulary scores (0.65). The first gold-cluster member remains an exploratory anchor, not a confirmatory input.

## Execution integrity and effective sample size

- n_queries: 341
- n_source_clusters: 78 (CSC 21, MLA 24, TAP 33)
- Confidence intervals: 95% cluster bootstrap (2000 resamples; clusters, not queries)
- Feature distinctness floor: 0.90
- Distinct retrieval feature vectors: 323/341 (0.9472)
- Runtime errors: 0
- Forbidden answer/gold fields: 0

## Empty retrievals

zero_hit_queries: none; zero_hit_rate: 0.0000.
A zero-hit packet is `absent` by construction of the BM25 `score > 0` filter, not by an observed evidence state. It must be explicitly flagged to annotators; it must not be presented as evidence that relevant sources do not exist.

## Retrieval results

Values are estimates with cluster-bootstrap 95% CIs. Precision@10 is retained only as a raw diagnostic with its attainable ceiling.

| Population | n | Recall@10 | Precision@|rel| | Success@1 | Success@3 | Success@5 | Success@10 | MRR | Raw P@10 (ceiling) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Including zero hits | 341 | 0.7683 [0.7006, 0.8333] | 0.5587 [0.4718, 0.6425] | 0.6979 [0.5929, 0.7988] | 0.8094 [0.7187, 0.8949] | 0.8680 [0.7938, 0.9341] | 0.9179 [0.8649, 0.9634] | 0.7673 [0.6812, 0.8519] | 0.1513 (0.1977) |
| Excluding zero hits | 341 | 0.7683 [0.7006, 0.8333] | 0.5587 [0.4718, 0.6425] | 0.6979 [0.5929, 0.7988] | 0.8094 [0.7187, 0.8949] | 0.8680 [0.7938, 0.9341] | 0.9179 [0.8649, 0.9634] | 0.7673 [0.6812, 0.8519] | 0.1513 (0.1977) |

### Per-task integrity

| Task | n_queries | n_source_clusters | distinct vectors / rows | rate | zero hits | zero-hit rate |
|---|---:|---:|---:|---:|---:|---:|
| CSC | 111 | 21 | 104 / 111 | 0.9369 | 0 | 0.0000 |
| MLA | 95 | 24 | 89 / 95 | 0.9368 | 0 | 0.0000 |
| TAP | 135 | 33 | 130 / 135 | 0.9630 | 0 | 0.0000 |

## Interpretation boundary

This validates question-conditioned local CSKG retrieval and option-blind evidence construction only. It does not test the support-state classifier, answer generation, routing AURC, harm, or oracle-gap recovery.

Gate C is an abstention-policy claim in this frozen policy: the structural policy never selects `vanilla_rag`, so it must not be described as evidence of routing among retrieval strategies.
