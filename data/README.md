# PX-034 data contracts

Raw benchmark downloads are not copied into this workspace. Keep source checkouts in `data/raw/` or an external path and generated annotations/splits in their named directories.

The derived `data/cticonnect/catalog.jsonl` was imported from the official CTIConnect v1.0.0 release and remains subject to its CC-BY-4.0 data license. It deliberately contains gold data and must never be passed to the detector. The current official release contains 1,859 internally validated rows; the paper/project page states 1,860. This one-row discrepancy is recorded in `data/cticonnect/import_summary.json`.

## Detector input (`detector_input.jsonl`)

One JSON object per query:

```json
{
  "query_id": "cticonnect:ATA:001",
  "question": "Question text",
  "task": "entity_attribution",
  "evidence": [
    {
      "source_id": "report-17",
      "title": "Report title",
      "source_type": "vendor_report",
      "text": "Retrieved passage only",
      "published_at": "2025-01-15T00:00:00Z",
      "retrieved_at": "2026-08-28T00:00:00Z",
      "retrieval_score": 0.82,
      "canonical_entities": ["APT29", "T1566"]
    }
  ]
}
```

Answer options, generated answers, gold answers, and support labels are forbidden in this file and rejected recursively by the loader.

## Annotation file (`support_state_annotations.jsonl`)

Labels live in a separate file so they cannot enter feature extraction:

```json
{"query_id":"cticonnect:ATA:001","task":"entity_attribution","support_state":"conflicting","annotator_1":"conflicting","annotator_2":"stale","adjudicated":true,"rationale":"Vendor attribution changed after later reporting."}
```

Allowed labels are `decisive`, `complementary`, `conflicting`, `stale`, `absent`, and `other`. `Other` is used only for taxonomy-coverage Gate A and is excluded from detector fitting/evaluation.

## Arm outcomes (`arm_outcomes.jsonl`)

Store one record per query, model, and arm. Required common fields are `query_id`, `model_id`, `model_tier`, `arm`, `task`, `correct`, `invalid`, `unsupported`, `latency_ms`, `input_tokens`, and `output_tokens`. Preserve raw outputs and evaluator provenance in separate immutable artifacts.

## CTIConnect retrieval traces

The official baseline prediction writer does not currently retain retrieval hits. PX-034 therefore consumes an explicit trace file:

```json
{"id":"ata-001","retrieval_strategy":"vanilla_rag","retrieved_context":[{"doc_id":"T1553.005","title":"Mark-of-the-Web Bypass","text":"...","score":0.82,"source_type":"mitre"}]}
```

For domain-specific strategies, also retain the transformed query or decomposed behaviors as trace metadata, but do not place generated candidate answers in `retrieved_context`. `make-detector-input` strips the trace to the detector contract and extracts canonical CTI identifiers from evidence text.

## MDS anchor manifest

Confirmatory CSKG execution requires an independently supplied anchor manifest:

```json
{"query_id":"csc-001","anchor_id":"BLOG-108","provenance":"author clarification or frozen reconstruction rule"}
```

The public v1.0.0 QA schema contains the complete gold `blog_ids` cluster but does not identify the single anchor report described in the paper. Using the first gold-cluster member is therefore permitted only with `--exploratory-first-source-anchor`; traces and run manifests are marked accordingly.
