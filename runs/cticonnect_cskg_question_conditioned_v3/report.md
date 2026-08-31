# PX-034 question-conditioned CTIConnect CSKG run (v3)

Status: **execution pass; exploratory retrieval result only**

This run uses question-conditioned CSKG retrieval: normalized BM25 scores from question tokens (0.10) are fused with anchor-vocabulary scores (0.90). The first gold-cluster member remains an exploratory anchor, not a confirmatory input.
The weight was selected by maximizing recall@10 among sweep candidates clearing the 0.90 detector-feature distinctness floor in every task; see `weight_sweep.md`.

## Execution integrity and effective sample size

- n_queries: 341
- n_source_clusters: 78 (CSC 21, MLA 24, TAP 33)
- Confidence intervals: 95% cluster bootstrap (2000 resamples; clusters, not queries)
- Feature distinctness floor: 0.90
- Distinct retrieval vectors: 321/341 (0.9413)
- Distinct detector feature vectors: 324/341 (0.9501)
- Fully collapsed source clusters on detector features: 0
- Feature dimensions varying within at least one cluster: 14/17
- Runtime errors: 0
- Forbidden answer/gold fields: 0
- Date audit: ISO 2291, normalized 106, unparsed 0, null 1013; unparsed rate 0.0000 (maximum 0.0000)

## Empty retrievals

zero_hit_queries: none; zero_hit_rate: 0.0000.
A zero-hit packet is `absent` by construction of the BM25 `score > 0` filter, not by an observed evidence state. It must be explicitly flagged to annotators; it must not be presented as evidence that relevant sources do not exist.

## Retrieval results

Values are estimates with cluster-bootstrap 95% CIs. Precision@10 is retained only as a raw diagnostic with its attainable ceiling.

| Population | n | Recall@10 | Precision@|rel| | Success@1 | Success@3 | Success@5 | Success@10 | MRR | Raw P@10 (ceiling) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All queries | 341 | 0.8226 [0.7540, 0.8866] | 0.5938 [0.4959, 0.6847] | 0.6979 [0.5857, 0.8048] | 0.8299 [0.7432, 0.9083] | 0.8827 [0.8116, 0.9440] | 0.9326 [0.8722, 0.9806] | 0.7732 [0.6865, 0.8579] | 0.1625 (0.1977) |

### Per-task integrity

| Task | n_queries | n_source_clusters | distinct vectors / rows | rate | zero hits | zero-hit rate |
|---|---:|---:|---:|---:|---:|---:|
| CSC | 111 | 21 | 104 / 111 | 0.9369 | 0 | 0.0000 |
| MLA | 95 | 24 | 90 / 95 | 0.9474 | 0 | 0.0000 |
| TAP | 135 | 33 | 130 / 135 | 0.9630 | 0 | 0.0000 |

Including and excluding zero hits are identical because this run has no zero-hit queries; one row is shown.

## Interpretation boundary

This validates question-conditioned local CSKG retrieval and option-blind evidence construction only. It does not test the support-state classifier, answer generation, routing AURC, harm, or oracle-gap recovery.

Gate C is an abstention-policy claim in this frozen policy: the structural policy never selects `vanilla_rag`, so it must not be described as evidence of routing among retrieval strategies.
