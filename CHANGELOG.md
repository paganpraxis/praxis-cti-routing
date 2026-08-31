# Changelog

## 2026-08-31 — PX-034 v3 closeout

- Publication-date feature failures changed from `246/341` v2 detector rows to `0/341`. Across the `3,410` v3 evidence items, the normalized detector input contains `2,291` already-ISO dates, `106` allowlist-normalized dates, `0` unparsed dates, and `1,013` null/blank dates; `dates_unparsed_rate=0.0000` against a frozen maximum of `0.0000`. Every non-null source value is preserved in `published_at_raw`.
- Degeneracy is now measured on the real 17-dimensional `extract_features()` output. Detector-feature distinctness is `324/341` (`0.9501`) overall: CSC `104/111` (`0.9369`), MLA `90/95` (`0.9474`), TAP `130/135` (`0.9630`). Among the `76` multi-query source clusters, fully collapsed clusters are `0`; `14/17` dimensions vary within a cluster, while `3/17` are constant (`evidence_count`, `source_type_count`, `source_concentration`). Retrieval-vector distinctness is separately reported as `321/341` (`0.9413`): CSC `104/111` (`0.9369`), MLA `89/95` (`0.9368`), TAP `128/135` (`0.9481`).
- The CSKG fusion changed from question/anchor weights `0.35/0.65` to `0.10/0.90`. The seven-point sweep (`0.0` through `0.6` by `0.1`) selected `0.10` by maximum recall@10 among weights clearing `0.90` detector distinctness in every task; the former `0.35` is retained as a diagnostic row.
- Relative to v2, recall@10 changed `0.7683 → 0.8226`, full-cluster recovery `0.6188 → 0.7126`, MRR `0.7673 → 0.7732`, precision@|rel| `0.5587 → 0.5938`, and raw precision@10 `0.1513 → 0.1625`. Success@3/5/10 changed `0.8094/0.8680/0.9179 → 0.8299/0.8827/0.9326`; success@1 remains `0.6979`, the mean precision@10 ceiling remains `0.1977`, and zero hits remain `0/341` (`0.0000`).
- Gate C was reconciled everywhere using the abstention-only decision: proposal H3/RQ3 now test state-conditioned abstention, comparison arm 5 changed from `structural_router` to `structural_abstention_policy`, and no retrieval-strategy-selection claim or secondary science was added.
- The v1 exploratory report is now marked superseded by v2. Identical including/excluding-zero-hit rows were collapsed to one row plus a note. The regression suite changed from `17` to `19` tests.

## 2026-08-31 — PX-034 integrity pass

- Question-conditioned CSKG retrieval replaced anchor-only ranking. The manifest now records question weight `0.35` and anchor weight `0.65`.
- Retrieval-feature distinctness changed from `78/341` (`0.2287`) to `323/341` (`0.9472`). Per task it changed from the anchor-only ceilings CSC `21/111` (`0.1892`), MLA `24/95` (`0.2526`), and TAP `33/135` (`0.2444`) to CSC `104/111` (`0.9369`), MLA `89/95` (`0.9368`), and TAP `130/135` (`0.9630`). The declared per-task floor is `0.90`.
- Effective sample size is now explicit: `n_queries=341`, `n_source_clusters=78`; CSC `111/21`, MLA `95/24`, TAP `135/33`. All reported 95% CIs use `2,000` source-cluster bootstrap resamples.
- Question conditioning removed the four former empty results (`csc-035`–`csc-038`): `zero_hit_queries` changed from `4` to `0`, and `zero_hit_rate` from `0.0117` to `0.0000`. Including- and excluding-zero-hit metrics remain separately reported, and future filter-empty packets are flagged as construction-induced `absent` cases for annotators.
- Corrected full-run retrieval estimates: recall@10 `0.8138 → 0.7683`, full-cluster recovery `0.7009 → 0.6188`, MRR `0.7783 → 0.7673`, and raw precision@10 `0.1607 → 0.1513` (attainable mean ceiling `0.1977`). The primary precision diagnostic is now precision@|rel| `0.5587`; success@1/3/5/10 is `0.6979/0.8094/0.8680/0.9179`.
- Gate E changed from `0.09` to `0.375` of the oracle gap, the dimensionally correct `9/24` threshold.
- Gate C was renamed from a routing claim to an abstention-policy claim because the frozen structural policy selects no `vanilla_rag` route.
- The regression suite count reported for this run changed from `14` to `17` tests.
