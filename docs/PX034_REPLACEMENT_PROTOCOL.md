# PX-034 replacement experiment protocol

Status: implementation draft; **not frozen**

This experiment tests whether answer-independent evidence structure can predict the CTI support state and whether acting on that prediction improves risk at matched coverage.

## Construct and label mapping

The detector predicts five operational states:

| PX-034 state | CONFLICTS origin | CTI interpretation | Default action |
|---|---|---|---|
| decisive | no conflict | Relevant sources converge on one supported resolution | CTIConnect task-specific retrieval |
| complementary | complementary information | Sources contribute mutually compatible partial evidence | CTIConnect task-specific retrieval and synthesis |
| conflicting | conflicting opinions/outcomes plus factual source disagreement | Incompatible vendor attribution, taxonomy mapping, or entity resolution | abstain/review |
| stale | outdated information | Disagreement is explained by version or publication time | abstain/refresh |
| absent | no relevant sources | Retrieved evidence does not answer the question | closed-book |

CONFLICTS `misinformation` is not silently discarded. During annotation, a factually false CTI source is labeled `conflicting` when another relevant source contradicts it. A lone suspected false source is `other` pending a reliability adjudication rule. Gate A reports the `other` share before classifier fitting.

## Leakage boundary

Detector-visible fields are limited to the question, CTIConnect task category, and retrieved evidence metadata/text. Candidate answers, multiple-choice options, generated answers, correctness, gold answers, and support labels are prohibited. Labels are stored separately and joined only by the training/evaluation driver.

## Dataset and quarantine

- Primary: CTIConnect, using all available task records and its original source artifacts.
- Secondary: the frozen 2,500-row CTI-MCQ transfer set. The 500 rows used by the June PX-034 run are quarantined from confirmatory evaluation.
- Split unit: query. Stratify by task category and support state. Where multiple queries share a source cluster or synthesis report group, assign the entire group to one partition before freezing to prevent document leakage.
- Temporal results are reported separately using CTIConnect's dates; temporal evaluation does not replace the held-out split.

## Annotation

Two independent CTI-capable annotators label question-plus-evidence packets. They do not see answers or answer keys. Disagreements are reconciled, then reviewed by a third annotator. Preserve both initial labels, adjudicated label, rationale, timestamps, and evidence snapshot hash. Report raw agreement and Cohen's kappa before reconciliation.

The maximum `other` share and minimum per-class count remain TBD until a blinded annotation pilot. They must then be frozen before confirmatory annotation/evaluation.

## Arms

1. Closed-book.
2. Vanilla RAG with one fixed retrieval configuration.
3. Always domain-specific: EtR for entity linking, DtR for entity attribution, CSKG-guided for multi-document synthesis.
4. LLM self-prediction followed by state-conditioned generation.
5. Structural abstention policy: apply the frozen task-specific retrieval action for decisive/complementary predictions, use closed-book for absent predictions, and abstain for conflicting/stale/other predictions. This arm tests abstention, not retrieval-strategy selection.
6. Oracle state, reported only as an upper bound.

All answer-generation prompts, retrieval depth, evaluator versions, decoding settings, and model revisions must be identical where the arm definition permits.

## Confirmatory metrics

- Detection: macro-F1 (primary), accuracy, per-class precision/recall/F1, confusion matrix, and bootstrap confidence intervals. Every confidence interval resamples source clusters, never individual queries. Report both `n_queries` and `n_source_clusters` overall and per task.
- Routing: AURC (primary), risk at 25/50/75/100% matched coverage, selective accuracy, and abstention rate by task.
- Harm: among cases where closed-book is correct and vanilla RAG is wrong, the fraction diverted from vanilla RAG; also report beneficial-retrieval misroutes.
- Oracle recovery: `(router - vanilla) / (oracle - vanilla)` for expected-behavior adherence and answer score.
- Safety: invalid and unsupported claim rates.
- Efficiency: detector latency, tokens, and estimated cost per query at equal detector accuracy.

For MDS retrieval diagnostics, report precision@|rel| and success@1/3/5/10. Raw precision@10 may be retained only beside its attainable ceiling (0.20 for three-report gold clusters after excluding the anchor; 0.10 for two-report clusters). Report metrics both including and excluding zero-hit packets. A packet made empty by the BM25 `score > 0` filter is `absent` by construction of that filter, not by an observed evidence state, and must be flagged as such to annotators.

All metrics are reported overall, by CTIConnect task, model tier, model family, support state, and temporal half. Small cells receive intervals and are not used for positive claims.

## Required ablations

- Option-aware versus option-blind, with option-aware reported only as a leakage diagnostic.
- Remove retrieval-score geometry.
- Remove source/entity agreement.
- Remove temporal features.
- Remove textual overlap/conflict indicators.
- Disable abstention while keeping the frozen task-specific action, isolating the value of state-conditioned abstention.

## Gates before held-out access

The config currently leaves Gate A and Gate D thresholds null. That is deliberate: inventing them now would not make them principled. Set them from the blinded development/pilot procedure, record their justification, then hash the protocol, config, split manifest, prompts, label guide, and evaluator versions. The held-out runner should refuse to run unless the supplied freeze hash matches.

Gate B is the load-bearing stop: option-blind macro-F1 must exceed the majority baseline in both model families. Because the frozen `route_for_state` policy does not select vanilla RAG, Gate C is specifically an abstention-policy claim: its selective AURC must beat vanilla RAG and always-domain-specific in both tiers. It is not evidence that the policy routes among retrieval strategies. Gate E requires recovery of more than 0.375 of the oracle gap, because the published pipeline gain is +9 points out of a +24-point oracle opportunity (`9 / 24 = 0.375`). Gate S blocks any positive conclusion if routing increases invalid or unsupported output rate.

## Commands

```bash
python3 -m unittest discover -s tests -v
python3 -m px034 import-cticonnect --cticonnect-root /path/to/CTIConnect --output-dir data/cticonnect
python3 -m px034 make-detector-input --cticonnect-root /path/to/CTIConnect --retrieval-traces runs/cticonnect_retrieval/retrieval.jsonl --output data/detector_input.jsonl --summary runs/cticonnect_retrieval/import_summary.json
python3 -m px034 make-splits --annotations data/annotations/support_state_annotations.jsonl --output data/splits/px034_split.json
python3 -m px034 run-baseline --detector-input data/detector_input.jsonl --annotations data/annotations/support_state_annotations.jsonl --splits data/splits/px034_split.json --partition development --as-of 2026-08-28T00:00:00-04:00 --output-dir runs/px034_structural_baseline_dev
```

Do not invoke the held-out command until the protocol is frozen.

Retrieval instrumentation:

```bash
# Official dense vanilla retrieval (requires OPENAI_API_KEY and dependencies)
python3 -m px034 run-retrieval --cticonnect-root /path/to/CTIConnect --strategy vanilla --tasks rcm wim atd esd ata vca --top-k 5 --output-dir runs/cticonnect_vanilla

# Official EtR/DtR transformation prompts plus official dense retrieval
python3 -m px034 run-retrieval --cticonnect-root /path/to/CTIConnect --strategy etr --tasks rcm wim atd esd --transform-model gpt-4o --top-k 5 --output-dir runs/cticonnect_etr
python3 -m px034 run-retrieval --cticonnect-root /path/to/CTIConnect --strategy dtr --tasks ata vca --transform-model gpt-4o --top-k 5 --output-dir runs/cticonnect_dtr

# Offline query over the shipped CSKG, requiring a separately frozen anchor map
python3 -m px034 run-retrieval --cticonnect-root /path/to/CTIConnect --strategy cskg --tasks csc tap mla --anchor-manifest data/cticonnect/mds_anchors.jsonl --top-k 10 --output-dir runs/cticonnect_cskg
```

Each run writes `retrieval.jsonl` and `run_manifest.json`. Traces retain the raw or transformed retrieval query, per-query ranked hits, merged/deduplicated evidence, scores, matched CSKG tokens, canonical entities, publication dates, retrieval timestamp, per-stage latency, strategy, model identity, release hash, and upstream Git commit. Answer generation is deliberately not performed by the retrieval runner.

## CTIConnect release integration

The adapter validates every official task file against `data/manifest.json`, checks its SHA-256, validates record counts and task/category/evaluation types, and rejects duplicate query IDs. It imports gold-bearing records to a separate catalog and computes connected source-cluster IDs: records sharing a CVE/CWE/CAPEC/ATT&CK source or any vendor report are kept in the same split. This is stricter than query-only splitting and prevents the same report cluster from leaking across training and held-out partitions.

The official v1.0.0 manifest released 2026-05-28 is internally consistent at 1,859 rows: 1,139 entity-linking, 379 entity-attribution, and 341 synthesis. The paper and project page state 1,860; the released VCA file contains 219 rows rather than the stated 220. PX-034 records this discrepancy and uses the hash-verified official release as the executable substrate unless an updated version is frozen later.

The release contains a complete precomputed 321-report CSKG (`cskg/per_doc_entities.jsonl`) but no `baselines/cskg_guided/run.py`, despite its package documentation referencing one. Its public MDS records also provide the full gold report cluster but no explicit anchor field. PX-034 queries the shipped entity vocabulary with BM25Okapi-equivalent scoring and refuses confirmatory MDS execution without a separate anchor manifest.

Before detector construction, publication dates are normalized using only `%B %d, %Y`, `%b %d, %Y`, `%Y-%m-%d`, `%Y-%m`, and `%Y`. Month-only and year-only values use the first representable day produced by the declared parser; the original value remains in `published_at_raw`. Blank values are audited as null. Unrecognized nonblank values remain visible as raw strings, and detector-input construction fails if their rate exceeds the configured threshold (`0.0` for v3).
