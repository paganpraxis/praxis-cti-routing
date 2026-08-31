from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from px034.cticonnect import build_detector_inputs
from px034.cticonnect_execution import ShippedCSKGRetriever
from px034.detector import CentroidDetector
from px034.evaluation import evaluate_mds_retrieval
from px034.features import extract_features
from px034.metrics import aurc, classification_metrics, feature_degeneracy, harm_prevention_rate, oracle_gap_recovery
from px034.routing import route_for_state
from px034.retrieval import EntityBM25, Hit, execute_trace
from px034.schema import DetectorInput
from px034.split import freeze_hash, grouped_stratified_split, stratified_split


def detector_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": "q1",
        "question": "Which technique is associated with CVE-2024-0001?",
        "task": "entity_attribution",
        "evidence": [
            {
                "source_id": "vendor-a",
                "source_type": "vendor_report",
                "text": "CVE-2024-0001 maps to T1203.",
                "published_at": "2025-01-01T00:00:00Z",
                "retrieval_score": 0.9,
                "canonical_entities": ["CVE-2024-0001", "T1203"],
            },
            {
                "source_id": "attack",
                "source_type": "attack",
                "text": "T1203 is exploitation for client execution.",
                "published_at": "2025-02-01T00:00:00Z",
                "retrieval_score": 0.7,
                "canonical_entities": ["T1203"],
            },
        ],
    }
    row.update(updates)
    return row


class SchemaTests(unittest.TestCase):
    def test_option_blind_row_is_accepted(self) -> None:
        parsed = DetectorInput.from_dict(detector_row())
        self.assertEqual(parsed.query_id, "q1")

    def test_answer_options_are_rejected_at_any_depth(self) -> None:
        row = detector_row(metadata={"answer_options": ["A", "B"]})
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            DetectorInput.from_dict(row)

    def test_gold_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DetectorInput.from_dict(detector_row(gold_label="decisive"))


class FeatureTests(unittest.TestCase):
    def test_structural_features_are_deterministic(self) -> None:
        row = DetectorInput.from_dict(detector_row())
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        left = extract_features(row, as_of)
        right = extract_features(row, as_of)
        self.assertEqual(left, right)
        self.assertEqual(left["evidence_count"], 2.0)
        self.assertGreater(left["cti_id_pair_agreement"], 0.0)
        self.assertAlmostEqual(left["score_top2_margin"], 0.2)


class DetectorTests(unittest.TestCase):
    def test_centroid_detector_separates_simple_classes(self) -> None:
        model = CentroidDetector.fit(
            [{"count": 0.0}, {"count": 0.1}, {"count": 9.9}, {"count": 10.0}],
            ["absent", "absent", "decisive", "decisive"],
        )
        self.assertEqual(model.predict_one({"count": 0.2})[0], "absent")
        self.assertEqual(model.predict_one({"count": 9.8})[0], "decisive")


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self) -> None:
        rows = [
            {"query_id": f"q{i}", "task": "entity_linking", "support_state": "decisive" if i < 5 else "absent"}
            for i in range(10)
        ]
        first = stratified_split(rows, 34)
        second = stratified_split(rows, 34)
        self.assertEqual(first, second)
        sets = [set(first[name]) for name in ("train", "development", "heldout")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, sets)), 10)
        self.assertEqual(freeze_hash(first), freeze_hash(second))

    def test_grouped_split_keeps_shared_sources_together(self) -> None:
        rows = [
            {"query_id": "q1", "task": "tap", "support_state": "decisive", "source_cluster_id": "source:BLOG-1"},
            {"query_id": "q2", "task": "tap", "support_state": "conflicting", "source_cluster_id": "source:BLOG-1"},
            {"query_id": "q3", "task": "tap", "support_state": "decisive", "source_cluster_id": "source:BLOG-2"},
        ]
        split = grouped_stratified_split(rows, 34)
        partitions = {query_id: name for name, ids in split.items() for query_id in ids}
        self.assertEqual(partitions["q1"], partitions["q2"])


class CTIConnectAdapterTests(unittest.TestCase):
    def test_retrieval_trace_is_normalized_without_catalog_gold(self) -> None:
        catalog = [
            {
                "query_id": "ata-001",
                "question": "What technique is described?",
                "task": "ata",
                "category": "entity_attribution",
                "answer": "T1203",
                "ground_truth": {"target_ids": ["T1203"]},
            }
        ]
        traces = [
            {
                "id": "ata-001",
                "retrieved_context": [
                    {"doc_id": "T1203", "title": "Exploitation", "contents": "Technique T1203", "score": 0.8}
                ],
            }
        ]
        rows, summary = build_detector_inputs(catalog, traces)
        self.assertEqual(summary["detector_inputs"], 1)
        self.assertEqual(rows[0]["task"], "entity_attribution")
        self.assertNotIn("answer", rows[0])
        self.assertNotIn("ground_truth", rows[0])
        self.assertEqual(rows[0]["evidence"][0]["canonical_entities"], ["T1203"])

    def test_unknown_retrieval_ids_are_reported(self) -> None:
        rows, summary = build_detector_inputs([], [{"id": "unknown", "retrieved_context": []}])
        self.assertEqual(rows, [])
        self.assertEqual(summary["unknown_retrieval_queries"], ["unknown"])


class RetrievalInstrumentationTests(unittest.TestCase):
    def test_cskg_queries_sharing_anchor_produce_different_hit_lists(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cskg").mkdir()
            (root / "cskg" / "per_doc_entities.jsonl").write_text(
                "\n".join(
                    [
                        '{"doc_id":"ANCHOR","entity_surfaces":["APT29","malware"]}',
                        '{"doc_id":"COZY","entity_surfaces":["APT29","Cozy Bear","malware"]}',
                        '{"doc_id":"WELLMESS","entity_surfaces":["APT29","WellMess","malware"]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            retriever = ShippedCSKGRetriever(
                root,
                {"q1": "ANCHOR", "q2": "ANCHOR"},
                {},
            )
            retriever.set_query("q1")
            cozy = retriever.retrieve("Which report discusses Cozy Bear?", "csc", 2)
            retriever.set_query("q2")
            wellmess = retriever.retrieve("Which report discusses WellMess?", "csc", 2)
            self.assertNotEqual(
                [hit.source_id for hit in cozy],
                [hit.source_id for hit in wellmess],
            )

    def test_multi_query_trace_preserves_events_and_deduplicates_context(self) -> None:
        class FakeRetriever:
            def retrieve(self, query: str, task: str, k: int) -> list[Hit]:
                score = 0.9 if query == "behavior one" else 0.8
                return [Hit("T1203", "Exploitation", "T1203 evidence", score, "mitre")]

        def transform(question: str, task: str):
            return ["behavior one", "behavior two"], {"kind": "fake_decomposition", "usage": {"total_tokens": 5}}

        trace = execute_trace(
            {"query_id": "ata-001", "question": "question", "task": "ata", "category": "entity_attribution"},
            "dtr",
            FakeRetriever(),
            5,
            transform,
        )
        self.assertEqual(len(trace["retrieval_events"]), 2)
        self.assertEqual(len(trace["retrieved_context"]), 1)
        self.assertEqual(trace["retrieved_context"][0]["retrieval_score"], 0.9)
        self.assertIn("retrieved_at", trace)

    def test_entity_bm25_prefers_distinctive_overlap(self) -> None:
        index = EntityBM25(
            {
                "BLOG-A": ["APT29", "Cozy Bear", "WellMess"],
                "BLOG-B": ["APT29", "Sliver"],
                "BLOG-C": ["LockBit", "Ransomware"],
            }
        )
        hits = index.search(["Cozy Bear", "WellMess"], 2)
        self.assertEqual(hits[0][0], "BLOG-A")
        self.assertNotIn("BLOG-C", [row[0] for row in hits])


class MetricAndRoutingTests(unittest.TestCase):
    def test_mds_summary_reports_queries_and_cluster_bootstrap_unit(self) -> None:
        catalog = [
            {
                "query_id": "csc-001",
                "source": {"blog_ids": ["BLOG-A", "BLOG-B", "BLOG-C"]},
            },
            {
                "query_id": "csc-002",
                "source": {"blog_ids": ["BLOG-A", "BLOG-B", "BLOG-C"]},
            },
        ]
        traces = [
            {
                "id": query_id,
                "task": "csc",
                "anchor_id": "BLOG-A",
                "anchor_policy": "test",
                "retrieved_context": [{"source_id": hit, "retrieval_score": score}],
                "latency_ms": 1.0,
            }
            for query_id, hit, score in (("csc-001", "BLOG-B", 0.8), ("csc-002", "BLOG-C", 0.7))
        ]
        summary = evaluate_mds_retrieval(catalog, traces, bootstrap_samples=20, feature_distinct_floor=0.0)
        self.assertEqual(summary["n_queries"], 2)
        self.assertEqual(summary["n_source_clusters"], 1)
        self.assertEqual(summary["bootstrap"]["unit"], "source_cluster")
        metrics = summary["metrics_including_zero_hits"]["metrics"]
        self.assertEqual(metrics["precision_at_rel"]["estimate"], 0.5)
        self.assertEqual(metrics["success_at_1"]["estimate"], 1.0)
        self.assertEqual(metrics["precision_at_10_ceiling"]["estimate"], 0.2)

    def test_mds_summary_flags_and_excludes_zero_hit_queries(self) -> None:
        catalog = [{"query_id": "csc-035", "source": {"blog_ids": ["BLOG-278", "BLOG-X"]}}]
        traces = [
            {
                "id": "csc-035",
                "task": "csc",
                "anchor_id": "BLOG-278",
                "anchor_policy": "test",
                "retrieved_context": [],
                "latency_ms": 1.0,
            }
        ]
        summary = evaluate_mds_retrieval(catalog, traces, bootstrap_samples=20, feature_distinct_floor=0.0)
        self.assertEqual(summary["zero_hit_queries"], ["csc-035"])
        self.assertEqual(summary["zero_hit_rate"], 1.0)
        self.assertEqual(summary["metrics_excluding_zero_hits"]["n_queries"], 0)

    def test_metrics(self) -> None:
        metrics = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertLess(aurc([True, False, True], [0.9, 0.1, 0.8]), 0.5)
        self.assertAlmostEqual(oracle_gap_recovery(0.5, 0.7, 0.9), 0.5)
        degeneracy = feature_degeneracy(
            [
                {"task": "csc", "features": {"x": 1.0}},
                {"task": "csc", "features": {"x": 1.0}},
                {"task": "csc", "features": {"x": 2.0}},
            ]
        )
        self.assertEqual(degeneracy["csc"]["distinct_feature_vectors"], 2)
        self.assertAlmostEqual(degeneracy["csc"]["distinct_feature_vector_rate"], 2 / 3)

    def test_harm_prevention(self) -> None:
        rows = [
            {"closed_book_correct": True, "vanilla_rag_correct": False, "route": "closed_book"},
            {"closed_book_correct": True, "vanilla_rag_correct": False, "route": "vanilla_rag"},
        ]
        self.assertEqual(harm_prevention_rate(rows), 0.5)

    def test_route_policy(self) -> None:
        self.assertEqual(route_for_state("decisive", "entity_linking")["strategy"], "extract_then_retrieve")
        self.assertEqual(route_for_state("conflicting", "entity_linking")["route"], "abstain")
        self.assertEqual(route_for_state("absent", "entity_linking")["route"], "closed_book")


if __name__ == "__main__":
    unittest.main()
