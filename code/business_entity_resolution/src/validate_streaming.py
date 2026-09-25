"""Exact bounded-memory validation of both outputs, including target-ID existence."""

import argparse, csv, json, itertools
from pathlib import Path
import numpy as np


def encoded(s):
    prefix, number = s.split("-")
    if (
        prefix not in ("S2", "S3")
        or not number.isdigit()
        or not number.isascii()
        or len(number) > 18
    ):
        raise ValueError("Unexpected target ID format: " + s)
    # Prefix a sentinel digit so differently zero-padded IDs remain distinct.
    return ((prefix == "S3") << 63) | int("1" + number)


def validate(root, output):
    parts = []
    for src in (2, 3):
        with open(root / f"test_source{src}.tsv", encoding="utf-8") as f:
            next(f)
            part = np.fromiter(
                (encoded(line.split("\t", 1)[0]) for line in f if line.strip()),
                dtype=np.uint64,
            )
        parts.append(part)
    valid = np.concatenate(parts)
    del parts
    valid.sort()
    references = candidates = matches = singletons = 0
    seen_references = set()
    with open(root / "test_source1.tsv", encoding="utf-8", newline="") as sf, open(
        output / "matching_results.tsv", encoding="utf-8", newline=""
    ) as mf, open(output / "candidate_pairs.tsv", encoding="utf-8", newline="") as cf:
        source = csv.DictReader(sf, delimiter="\t")
        match = csv.reader(mf, delimiter="\t")
        cand = csv.reader(cf, delimiter="\t")
        assert next(match) == ["source1_entity_id", "matched_entity_ids"]
        assert next(cand) == ["source1_entity_id", "candidate_entity_ids"]
        for s, m, c in itertools.zip_longest(source, match, cand):
            assert (
                s is not None and m is not None and c is not None
            ), "Row count mismatch"
            assert (
                len(m) == len(c) == 2 and s["entity_id"] == m[0] == c[0]
            ), "Row/ID mismatch"
            assert m[0] not in seen_references, "Duplicate reference ID"
            seen_references.add(m[0])
            mids = m[1].split(",") if m[1] else []
            cids = c[1].split(",") if c[1] else []
            assert len(mids) == len(set(mids)) and len(cids) == len(
                set(cids)
            ), "Duplicate target IDs"
            assert set(mids) <= set(cids), "Match absent from candidate set"
            if cids:
                vals = np.asarray([encoded(v) for v in cids], dtype=np.uint64)
                pos = np.searchsorted(valid, vals)
                assert np.all(pos < len(valid)) and np.all(
                    valid[pos] == vals
                ), "Unknown target ID"
            references += 1
            candidates += len(cids)
            matches += len(mids)
            singletons += int(not mids)
            if references % 100000 == 0:
                print(f"Validated {references:,} reference rows", flush=True)
    result = {
        "status": "PASS",
        "references": references,
        "candidate_pairs": candidates,
        "matched_pairs": matches,
        "singletons": singletons,
        "checks": [
            "exact headers",
            "one row per input reference in original order",
            "no duplicate reference IDs across the full output",
            "no duplicate target IDs",
            "every target ID exists",
            "matches subset of candidates",
        ],
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    result = validate(a.test_dir, a.output)
    a.report.write_text(json.dumps(result, indent=2))
