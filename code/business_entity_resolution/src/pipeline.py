"""Offline, disk-backed business entity resolution. MIT licensed original code."""

import argparse, csv, hashlib, json, re, time, unicodedata, mmap
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import orjson
from rapidfuzz import fuzz, process
import heapq
from lightgbm import LGBMClassifier, Booster

DT = np.dtype([("key", "<u8"), ("row", "<u4")])
K = 7
STOP = set(
    "the and private pvt limited ltd inc incorporated corporation corp llc llp company co".split()
)
REPLACE = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "drive": "dr",
    "lane": "ln",
    "boulevard": "blvd",
    "suite": "ste",
    "apartment": "apt",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def norm(s):
    s = (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .replace("&", " and ")
    )
    return " ".join(re.findall(r"[a-z0-9]+", s))


def record(row):
    name = norm(row["business_name"])
    addr = " ".join(REPLACE.get(t, t) for t in norm(row["business_address"]).split())
    core = " ".join(t for t in name.split() if t not in STOP) or name
    return [row["entity_id"], name, addr, norm(row["country"]), core]


def keys(r):
    _, n, a, c, b = r
    compact = b.replace(" ", "")
    nums = re.findall(r"\d+", a)
    vals = [
        b,
        " ".join(sorted(b.split())),
        a,
        compact[:9],
        compact[-9:],
        (nums[0] + "|" + compact[:5]) if nums else "",
        (" ".join(a.split()[:3])),
    ]
    return [
        (
            int.from_bytes(
                hashlib.blake2b(
                    (str(i) + "|" + c + "|" + v).encode(), digest_size=8
                ).digest(),
                "little",
            )
            if v
            else 0
        )
        for i, v in enumerate(vals)
    ]


def rows(root, split, source):
    with open(
        root / "dataset" / split / f"{split}_source{source}.tsv",
        encoding="utf-8",
        newline="",
    ) as f:
        yield from csv.DictReader(f, delimiter="\t")


def tokenkeys(r, field):
    return {
        int.from_bytes(
            hashlib.blake2b((r[3] + "|" + t).encode(), digest_size=8).digest(), "little"
        )
        for t in r[field].split()
        if len(t) >= 3 and t not in STOP
    }


def build_tokens(folder):
    if (folder / "tokens_complete.json").exists():
        return
    handles = [open(folder / f"tokens{field}.bin", "wb") for field in (4, 2)]
    buffers = [[], []]
    count = 0
    with open(folder / "records.jsonl", "rb") as f:
        for rid, line in enumerate(f):
            r = json.loads(line)
            for j, field in enumerate((4, 2)):
                buffers[j].extend((k, rid) for k in tokenkeys(r, field))
            count += 1
            if count % 50000 == 0:
                for h, b in zip(handles, buffers):
                    np.asarray(b, dtype=DT).tofile(h)
                    b.clear()
            if count % 1000000 == 0:
                log(f"{folder.name}: token index {count:,} targets")
    for h, b in zip(handles, buffers):
        np.asarray(b, dtype=DT).tofile(h)
        h.close()
    for field in (4, 2):
        log(f"{folder.name}: sorting token field {field}")
        a = np.memmap(folder / f"tokens{field}.bin", dtype=DT, mode="r+")
        a.sort(order="key", kind="quicksort")
        a.flush()
        np.asarray(a["key"]).tofile(folder / f"tokens{field}_lookup.bin")
        del a
    (folder / "tokens_complete.json").write_text(json.dumps({"records": count}))


def build(root, split, cache):
    folder = cache / split
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        return
    handles = [open(folder / f"key{i}.bin", "wb") for i in range(K)]
    offsets = open(folder / "offsets.bin", "wb")
    data = open(folder / "records.jsonl", "wb")
    batch = []
    off = []
    count = 0

    def flush():
        arr = np.asarray(batch, dtype=np.uint64)
        for j, h in enumerate(handles):
            out = np.empty(len(batch), dtype=DT)
            out["key"] = arr[:, j]
            out["row"] = np.arange(count - len(batch), count, dtype=np.uint32)
            out.tofile(h)
        np.asarray(off, dtype=np.uint64).tofile(offsets)
        batch.clear()
        off.clear()

    for src in [2, 3]:
        for row in rows(root, split, src):
            r = record(row)
            off.append(data.tell())
            data.write((json.dumps(r, ensure_ascii=True) + "\n").encode())
            batch.append(keys(r))
            count += 1
            if len(batch) == 50000:
                flush()
            if count % 500000 == 0:
                log(f"{split}: indexed {count:,} target records")
    if batch:
        flush()
    for h in handles:
        h.close()
    offsets.close()
    data.close()
    for j in range(K):
        log(f"{split}: sorting index {j+1}/{K}")
        a = np.memmap(folder / f"key{j}.bin", dtype=DT, mode="r+")
        a.sort(order="key", kind="quicksort")
        a.flush()
        del a
    (folder / "complete.json").write_text(json.dumps({"records": count}))


class Index:
    def __init__(self, folder, posting_limit=2000):
        self.posting_limit = posting_limit
        self.arr = [
            np.memmap(folder / f"key{i}.bin", dtype=DT, mode="r") for i in range(K)
        ]
        # Contiguous keys prevent NumPy from copying a strided 80 MB column on every search.
        self.lookup = []
        for j, a in enumerate(self.arr):
            path = folder / f"lookup{j}.bin"
            if not path.exists():
                np.asarray(a["key"]).tofile(path)
            self.lookup.append(np.memmap(path, dtype="<u8", mode="r"))
        self.off = np.asarray(np.memmap(folder / "offsets.bin", dtype="<u8", mode="r"))
        self.f = open(folder / "records.jsonl", "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.tokens = []
        if (folder / "tokens_complete.json").exists():
            for field in (4, 2):
                self.tokens.append(
                    (
                        field,
                        np.memmap(folder / f"tokens{field}.bin", dtype=DT, mode="r"),
                        np.memmap(
                            folder / f"tokens{field}_lookup.bin", dtype="<u8", mode="r"
                        ),
                    )
                )

    def get(self, i):
        start = int(self.off[i])
        end = int(self.off[i + 1]) if i + 1 < len(self.off) else len(self.mm)
        return orjson.loads(self.mm[start:end])

    def candidates(self, r):
        found = set()
        for j, key in enumerate(keys(r)):
            if not key:
                continue
            k = np.uint64(key)
            lo = int(np.searchsorted(self.lookup[j], k, "left"))
            hi = int(np.searchsorted(self.lookup[j], k, "right"))
            if hi - lo <= self.posting_limit:
                found.update(map(int, np.asarray(self.arr[j]["row"][lo:hi])))
        for field, a, lookup in self.tokens:
            posts = []
            for key in tokenkeys(r, field):
                k = np.uint64(key)
                lo = int(np.searchsorted(lookup, k, "left"))
                hi = int(np.searchsorted(lookup, k, "right"))
                if 0 < hi - lo <= self.posting_limit:
                    posts.append((hi - lo, lo, hi))
            for _, lo, hi in sorted(posts)[:4]:
                found.update(map(int, np.asarray(a["row"][lo:hi])))
        cs = [self.get(i) for i in sorted(found)]
        if len(cs) <= 160:
            return cs
        name = [b[4] for b in cs]
        addr = [b[2] for b in cs]

        def similarities(query, choices):
            return (
                process.cdist(
                    [query], choices, scorer=fuzz.ratio, dtype=np.float64, workers=1
                )[0]
                + process.cdist(
                    [query],
                    choices,
                    scorer=fuzz.token_set_ratio,
                    dtype=np.float64,
                    workers=1,
                )[0]
            )

        ns = similarities(r[4], name)
        ads = similarities(r[2], addr)
        chosen = set()
        for vals, num in [(ns + ads, 80), (ns, 40), (ads, 40)]:
            scores = vals.tolist()
            chosen.update(
                heapq.nsmallest(
                    num, range(len(cs)), key=lambda i: (-scores[i], cs[i][0])
                )
            )
        return [cs[i] for i in sorted(chosen)]


def features(a, b):
    out = []
    for x, y in [(a[1], b[1]), (a[2], b[2]), (a[4], b[4])]:
        sx, sy = set(x.split()), set(y.split())
        shared = len(sx & sy)
        out += [
            float(bool(x) and x == y),
            fuzz.ratio(x, y) / 100,
            fuzz.token_sort_ratio(x, y) / 100,
            fuzz.token_set_ratio(x, y) / 100,
            fuzz.partial_ratio(x, y) / 100,
            shared / max(1, len(sx | sy)),
            shared / max(1, min(len(sx), len(sy))),
            min(len(x), len(y)) / max(1, len(x), len(y)),
        ]
    nx, ny = set(re.findall(r"\d+", a[2])), set(re.findall(r"\d+", b[2]))
    out += [
        len(nx & ny) / max(1, len(nx | ny)),
        float(bool(nx & ny)),
        float(bool(nx and ny) and not nx & ny),
        float(a[3] == b[3]),
        float(b[0].startswith("S3-")),
    ]
    # Numeric normalization handles zero-padded municipal numbers without changing IDs.
    ax = [str(int(v)) for v in re.findall(r"\d+", a[2])]
    bx = [str(int(v)) for v in re.findall(r"\d+", b[2])]
    na, nb = set(ax), set(bx)
    out += [
        len(na & nb) / max(1, len(na | nb)),
        len(na & nb) / max(1, min(len(na), len(nb))),
        float(bool(ax and bx) and ax[0] == bx[0]),
        float(bool(na and nb) and not na & nb),
        len(na - nb),
        len(nb - na),
        float(not b[1]),
        float(not b[2]),
    ]
    for x, y in [(a[4], b[4]), (a[2], b[2])]:
        xx = re.sub(r"\d+", lambda m: str(int(m[0])), x)
        yy = re.sub(r"\d+", lambda m: str(int(m[0])), y)
        out += [
            fuzz.ratio(xx.replace(" ", ""), yy.replace(" ", "")) / 100,
            fuzz.token_sort_ratio(xx, yy) / 100,
            fuzz.token_set_ratio(xx, yy) / 100,
            fuzz.ratio(re.sub(r"\d+", "", x), re.sub(r"\d+", "", y)) / 100,
        ]
    return out


def score(truth, pred):
    if not truth:
        return float(not pred)
    return 1.25 * len(truth & pred) / (0.25 * len(truth) + len(pred)) if pred else 0.0


def train(root, cache, out, sample):
    if sample != 40000:
        raise ValueError("The reproducible tuning protocol uses --sample 40000")
    from train_tuned import run

    run(root, cache, out, cache / "tuning_v2")


def worker_init(folder, model_path, threshold):
    global WORK_INDEX, WORK_MODEL, WORK_THRESHOLD
    WORK_INDEX = Index(Path(folder))
    WORK_MODEL = Booster(model_file=str(model_path))
    WORK_THRESHOLD = threshold


def worker_score(batch):
    xx = []
    meta = []
    for r in batch:
        cs = WORK_INDEX.candidates(r)
        start = len(xx)
        xx.extend(features(r, b) for b in cs)
        meta.append((r, [b[0] for b in cs], start, len(xx)))
    probs = (
        WORK_MODEL.predict(np.asarray(xx, dtype=np.float32), num_threads=1)
        if xx
        else []
    )
    return [
        (r[0], r[3], ids, [v for v, p in zip(ids, probs[s:e]) if p >= WORK_THRESHOLD])
        for r, ids, s, e in meta
    ]


def resume_outputs(source, output):
    count = pairs = matches = empty = 0
    countries = Counter()
    with open(output / "matching_results.tsv", "r+b") as mf, open(
        output / "candidate_pairs.tsv", "r+b"
    ) as cf:
        assert mf.readline() == b"source1_entity_id\tmatched_entity_ids\n"
        assert cf.readline() == b"source1_entity_id\tcandidate_entity_ids\n"
        mpos, cpos = mf.tell(), cf.tell()
        while True:
            m, c = mf.readline(), cf.readline()
            if not m or not c or not m.endswith(b"\n") or not c.endswith(b"\n"):
                break
            sid, mids = m.rstrip(b"\n").split(b"\t")
            cid, cids = c.rstrip(b"\n").split(b"\t")
            row = next(source)
            assert (
                sid == cid == row["entity_id"].encode()
            ), "Existing output is not an aligned input prefix"
            count += 1
            pairs += cids.count(b",") + 1 if cids else 0
            matches += mids.count(b",") + 1 if mids else 0
            empty += int(not mids)
            countries[norm(row["country"])] += 1
            mpos, cpos = mf.tell(), cf.tell()
        mf.truncate(mpos)
        cf.truncate(cpos)
    log(f"Resuming after {count:,} verified complete reference rows")
    return count, pairs, matches, empty, countries


def predict(root, cache, out, output, workers=6, resume=False):
    build(root, "test", cache)
    build_tokens(cache / "test")
    idx = Index(cache / "test")
    del idx  # Create shared contiguous indexes before workers start.
    threshold = json.loads((out / "metrics.json").read_text())["threshold"]
    output.mkdir(exist_ok=True, parents=True)
    count = pairs = matches = empty = 0
    countries = Counter()
    source = iter(rows(root, "test", 1))
    if resume:
        count, pairs, matches, empty, countries = resume_outputs(source, output)
    mode = "a" if resume else "w"
    mf = open(output / "matching_results.tsv", mode, encoding="utf-8", newline="")
    cf = open(output / "candidate_pairs.tsv", mode, encoding="utf-8", newline="")
    if not resume:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")

    def consume(results):
        nonlocal count, pairs, matches, empty
        for sid, country, ids, pred in results:
            mf.write(sid + "\t" + ",".join(pred) + "\n")
            cf.write(sid + "\t" + ",".join(ids) + "\n")
            count += 1
            pairs += len(ids)
            matches += len(pred)
            empty += int(not pred)
            countries[country] += 1
        mf.flush()
        cf.flush()
        if count % 10000 == 0:
            log(
                f"predict: {count:,} references, {pairs:,} candidates, {matches:,} matches"
            )

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=worker_init,
        initargs=(str(cache / "test"), str(out / "model.txt"), threshold),
    ) as pool:
        pending = []
        batch = []
        for row in source:
            batch.append(record(row))
            if len(batch) == 500:
                pending.append(pool.submit(worker_score, batch))
                batch = []
                if len(pending) >= workers * 2:
                    consume(pending.pop(0).result())
        if batch:
            pending.append(pool.submit(worker_score, batch))
        for future in pending:
            consume(future.result())
    mf.close()
    cf.close()
    report = {
        "references": count,
        "candidate_pairs": pairs,
        "matched_pairs": matches,
        "singletons": empty,
        "countries": dict(countries),
    }
    (out / "prediction_summary.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(report))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--sample", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume aligned output files from this exact model, threshold and dataset",
    )
    ap.add_argument("--stage", choices=["train", "predict", "all"], default="all")
    args = ap.parse_args()
    args.cache.mkdir(exist_ok=True, parents=True)
    args.artifacts.mkdir(exist_ok=True, parents=True)
    if args.stage in ["train", "all"]:
        train(args.data_root, args.cache, args.artifacts, args.sample)
    if args.stage in ["predict", "all"]:
        predict(
            args.data_root,
            args.cache,
            args.artifacts,
            args.output,
            args.workers,
            args.resume,
        )
