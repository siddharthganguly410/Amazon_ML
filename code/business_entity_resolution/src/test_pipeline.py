import csv, tempfile, unittest
from pathlib import Path
from pipeline import record, keys, build, build_tokens, Index, score, features
from validate_streaming import validate, encoded
from validate_official_batched import run


class PipelineTests(unittest.TestCase):
    def test_zero_padded_address_numbers_and_id_independence(self):
        a = ["S1-1", "cafe", "12 rue victor hugo", "france", "cafe"]
        b = ["S2-999", "cafe", "0012 rue victor hugo", "france", "cafe"]
        values = features(a, b)
        self.assertEqual(values[29], 1.0)
        self.assertEqual(values[31], 1.0)
        self.assertEqual(values[32], 0.0)
        b[0] = "S2-00000001"
        self.assertEqual(values, features(a, b))
        b[2] = "19 rue victor hugo"
        self.assertEqual(features(a, b)[32], 1.0)

    def test_metric(self):
        self.assertEqual(score(set(), set()), 1)
        self.assertEqual(score(set(), {"x"}), 0)
        self.assertEqual(score({"a"}, set()), 0)
        self.assertAlmostEqual(score({"a", "b"}, {"a", "b", "c"}), 2.5 / 3.5)
        self.assertNotEqual(encoded("S2-1"), encoded("S2-01"))
        self.assertNotEqual(encoded("S2-1"), encoded("S3-1"))

    def test_disk_index_and_unseen_country(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "dataset/test").mkdir(parents=True)
            for source, name in [(2, "Cafe Lumiere Ltd"), (3, "Other Store")]:
                with open(
                    root / f"dataset/test/test_source{source}.tsv",
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(
                        ["entity_id", "business_name", "business_address", "country"]
                    )
                    w.writerow([f"S{source}-1", name, "12 Rue Victor Hugo", "France"])
            build(root, "test", root / "cache")
            build_tokens(root / "cache/test")
            idx = Index(root / "cache/test")
            q = record(
                dict(
                    entity_id="S1-1",
                    business_name="Café Lumière Limited",
                    business_address="12 rue Victor Hugo",
                    country="France",
                )
            )
            cs = idx.candidates(q)
            self.assertIn("S2-1", {r[0] for r in cs})
            self.assertEqual(len(features(q, cs[0])), 45)
            idx.mm.close()
            idx.f.close()
            idx.arr.clear()
            idx.lookup.clear()
            idx.tokens.clear()
            del idx.off
            (root / "dataset/test/test_source1.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tCafe\t12 Rue\tFrance\n"
            )
            output = root / "output"
            output.mkdir()
            (output / "matching_results.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n"
            )
            (output / "candidate_pairs.tsv").write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\n"
            )
            self.assertEqual(validate(root / "dataset/test", output)["status"], "PASS")
            self.assertEqual(run(root / "dataset/test", output)["status"], "PASS")
            (output / "candidate_pairs.tsv").write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-999\n"
            )
            with self.assertRaises(AssertionError):
                validate(root / "dataset/test", output)


if __name__ == "__main__":
    unittest.main()
