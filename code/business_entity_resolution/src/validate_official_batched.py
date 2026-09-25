"""Run the unmodified supplied validator in batches to bound its set/dict memory.

Use validate_streaming.py as well: it verifies global coverage and all target IDs.
"""

import argparse, contextlib, io, itertools, json, tempfile
from pathlib import Path
from official_validate_submission import validate


def run(root, output):
    batches = total = 0
    with tempfile.TemporaryDirectory() as tmp, open(
        root / "test_source1.tsv", encoding="utf-8"
    ) as sf, open(output / "matching_results.tsv", encoding="utf-8") as mf, open(
        output / "candidate_pairs.tsv", encoding="utf-8"
    ) as cf:
        tmp = Path(tmp)
        next(sf)
        mh = next(mf)
        ch = next(cf)
        records = itertools.zip_longest(sf, mf, cf)
        while True:
            chunk = list(itertools.islice(records, 25000))
            if not chunk:
                break
            with open(
                tmp / "test_source1.tsv", "w", encoding="utf-8", newline=""
            ) as a, open(
                tmp / "matches.tsv", "w", encoding="utf-8", newline=""
            ) as b, open(
                tmp / "candidates.tsv", "w", encoding="utf-8", newline=""
            ) as c:
                a.write("entity_id\n")
                b.write(mh)
                c.write(ch)
                for s, m, cs in chunk:
                    assert (
                        s is not None and m is not None and cs is not None
                    ), "Row count mismatch"
                    a.write(s.split("\t", 1)[0] + "\n")
                    b.write(m)
                    c.write(cs)
            with contextlib.redirect_stdout(io.StringIO()):
                errors, warnings = validate(
                    str(tmp / "matches.tsv"),
                    str(tmp / "candidates.tsv"),
                    str(tmp),
                    check_ids=False,
                )
            assert not errors, errors
            assert all("ID-existence check is OFF" in w for w in warnings), warnings
            batches += 1
            total += len(chunk)
            if batches % 10 == 0:
                print(f"Official validator: {total:,} rows passed", flush=True)
    result = {
        "status": "PASS",
        "batches": batches,
        "rows": total,
        "batch_size": 25000,
        "note": "Unmodified official validator checks both outputs in bounded-memory batches. Global coverage and exact target-ID existence checked separately by validate_streaming.py.",
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    a.report.write_text(json.dumps(run(a.test_dir, a.output), indent=2))
