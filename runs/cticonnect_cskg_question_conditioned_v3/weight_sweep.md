# CSKG question-weight sweep

The frozen selection rule is: choose the question weight with the highest overall recall@10 among candidates whose detector-feature distinctness is at least `0.90` in every task; break an exact recall tie toward the lower question weight. Anchor weight is `1 - question_weight`. The `0.35` row is retained as a v2 diagnostic and was not an additional sweep candidate.

| Question weight | Anchor weight | Detector distinct vectors | Overall distinctness | CSC | MLA | TAP | Recall@10 | Full-cluster recovery | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.00 | 1.00 | 320 / 341 | 0.9384 | 0.9099 | 0.9368 | 0.9630 | 0.8138 | 0.7009 | yes |
| **0.10** | **0.90** | **324 / 341** | **0.9501** | **0.9369** | **0.9474** | **0.9630** | **0.8226** | **0.7126** | **yes — selected** |
| 0.20 | 0.80 | 324 / 341 | 0.9501 | 0.9369 | 0.9474 | 0.9630 | 0.8196 | 0.7067 | yes |
| 0.30 | 0.70 | 324 / 341 | 0.9501 | 0.9369 | 0.9474 | 0.9630 | 0.7962 | 0.6598 | yes |
| 0.40 | 0.60 | 325 / 341 | 0.9531 | 0.9459 | 0.9474 | 0.9630 | 0.7331 | 0.5777 | yes |
| 0.50 | 0.50 | 325 / 341 | 0.9531 | 0.9459 | 0.9474 | 0.9630 | 0.6818 | 0.4956 | yes |
| 0.60 | 0.40 | 325 / 341 | 0.9531 | 0.9459 | 0.9474 | 0.9630 | 0.5557 | 0.3460 | yes |
| 0.35 (v2 diagnostic) | 0.65 | 324 / 341 | 0.9501 | 0.9369 | 0.9474 | 0.9630 | 0.7683 | 0.6188 | yes |

Weight `0.10` is frozen because it clears the detector-feature floor in all three tasks and maximizes recall@10 (`0.8226`) among the seven declared sweep candidates. It also improves full-cluster recovery to `0.7126`; retaining v2's `0.35` would sacrifice both retrieval measures without increasing detector distinctness.
