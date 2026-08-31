# Changelog

## 2026-08-31 — PX-034 integrity pass

- Question-conditioned CSKG retrieval replaced anchor-only ranking. The manifest now records question weight `0.35` and anchor weight `0.65`.
- Retrieval-feature distinctness changed from `78/341` (`0.2287`) to `323/341` (`0.9472`). Per task it changed from the anchor-only ceilings CSC `21/111` (`0.1892`), MLA `24/95` (`0.2526`), and TAP `33/135` (`0.2444`) to CSC `104/111` (`0.9369`), MLA `89/95` (`0.9368`), and TAP `130/135` (`0.9630`). The declared per-task floor is `0.90`.
- Effective sample size is now explicit: `n_queries=341`, `n_source_clusters=78`; CSC `111/21`, MLA `95/24`, TAP `135/33`. All reported 95% CIs use `2,000` source-cluster bootstrap resamples.
- Question conditioning removed the four former empty results (`csc-035`–`csc-038`): `zero_hit_queries` changed from `4` to `0`, and `zero_hit_rate` from `0.0117` to `0.0000`. Including- and excluding-zero-hit metrics remain separately reported, and future filter-empty packets are flagged as construction-induced `absent` cases for annotators.
- Corrected full-run retrieval estimates: recall@10 `0.8138 → 0.7683`, full-cluster recovery `0.7009 → 0.6188`, MRR `0.7783 → 0.7673`, and raw precision@10 `0.1607 → 0.1513` (attainable mean ceiling `0.1977`). The primary precision diagnostic is now precision@|rel| `0.5587`; success@1/3/5/10 is `0.6979/0.8094/0.8680/0.9179`.
- Gate E changed from `0.09` to `0.375` of the oracle gap, the dimensionally correct `9/24` threshold.
- Gate C was renamed from a routing claim to an abstention-policy claim because the frozen structural policy selects no `vanilla_rag` route.
- The regression suite count reported for this run changed from `14` to `17` tests.
