import csv, json, sys, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pipeline as p
from lightgbm import LGBMClassifier

ROOT = None
BASE = None
CACHE = None


def select():
    if (BASE / "selected.json").exists():
        return json.loads((BASE / "selected.json").read_text())
    dev = set()
    with open(ROOT / "dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            dev.add(r["source1_entity_id"])
            if len(dev) == 2000:
                break
    rng = np.random.default_rng(2026)
    old = []
    total = 0
    for r in p.rows(ROOT, "train", 1):
        if r["entity_id"] in dev:
            continue
        total += 1
        if len(old) < 30000:
            old.append(r["entity_id"])
        else:
            j = int(rng.integers(total))
            if j < 30000:
                old[j] = r["entity_id"]
    excluded = dev | set(old)
    rng = np.random.default_rng(92027)
    chosen = []
    total = 0
    for r in p.rows(ROOT, "train", 1):
        if r["entity_id"] in excluded:
            continue
        total += 1
        if len(chosen) < 40000:
            chosen.append(p.record(r))
        else:
            j = int(rng.integers(total))
            if j < 40000:
                chosen[j] = p.record(r)
    rng.shuffle(chosen)
    want = {r[0] for r in chosen}
    truth = {}
    with open(ROOT / "dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["source1_entity_id"] in want:
                truth[r["source1_entity_id"]] = list(
                    filter(None, r["matched_entity_ids"].split(","))
                )
    result = [(r, truth[r[0]]) for r in chosen]
    (BASE / "selected.json").write_text(json.dumps(result))
    p.log(
        "Selected 40,000 new entities, excluding prior 30,000 and 2,000 development entities"
    )
    return result


def init(cache, base):
    global IDX, BASE
    BASE = Path(base)
    IDX = p.Index(Path(cache) / "train", posting_limit=5000)


def batch(args):
    number, items = args
    path = BASE / f"shard{number:04d}.npz"
    meta = BASE / f"shard{number:04d}.json"
    if path.exists() and meta.exists():
        return number, len(items)
    xx = []
    yy = []
    groups = []
    for r, truth in items:
        cs = IDX.candidates(r)
        gt = set(truth)
        start = len(xx)
        for b in cs:
            xx.append(p.features(r, b))
            yy.append(int(b[0] in gt))
        groups.append([r, truth, [b[0] for b in cs], start, len(xx)])
    np.savez(path, X=np.asarray(xx, dtype=np.float32), y=np.asarray(yy, dtype=np.uint8))
    meta.write_text(json.dumps(groups))
    return number, len(items)


def load():
    xs = []
    ys = []
    groups = []
    off = 0
    for meta in sorted(BASE.glob("shard*.json")):
        z = np.load(meta.with_suffix(".npz"))
        xs.append(z["X"])
        ys.append(z["y"])
        for r, gt, ids, s, e in json.loads(meta.read_text()):
            groups.append((r, gt, ids, s + off, e + off))
        off += len(z["y"])
    return np.concatenate(xs), np.concatenate(ys), groups


def evaluator(groups, y, probs):
    counts = np.array([e - s for _, _, _, s, e in groups])
    gids = np.repeat(np.arange(len(groups)), counts)
    truth = np.array([len(gt) for _, gt, _, _, _ in groups])
    blocked = np.bincount(gids, weights=y, minlength=len(groups))

    def evaluate(th):
        pred = probs >= th
        pc = np.bincount(gids, weights=pred, minlength=len(groups))
        tp = np.bincount(gids, weights=pred * y, minlength=len(groups))
        scores = np.divide(
            1.25 * tp,
            0.25 * truth + pc,
            out=np.zeros(len(groups)),
            where=(0.25 * truth + pc) > 0,
        )
        scores[truth == 0] = pc[truth == 0] == 0
        return {
            "macro_f0.5": float(scores.mean()),
            "pair_precision": float(tp.sum() / max(1, pc.sum())),
            "pair_recall": float(tp.sum() / max(1, truth.sum())),
            "blocking_recall": float(blocked.sum() / max(1, truth.sum())),
            "entities": len(groups),
        }

    return evaluate


def train():
    X, y, gs = load()
    assert len(gs) == 40000
    end = gs[32000][3]
    v_end = gs[36000][3]
    best = None
    trials = []
    for trees, leaves, rate in [(450, 31, 0.045), (700, 63, 0.04), (1000, 31, 0.035)]:
        conf = dict(
            n_estimators=trees,
            num_leaves=leaves,
            learning_rate=rate,
            min_child_samples=40,
            colsample_bytree=0.9,
            reg_lambda=2,
            n_jobs=8,
            random_state=2026,
            verbosity=-1,
        )
        m = LGBMClassifier(**conf)
        p.log("Fitting " + str(conf))
        m.fit(X[:end], y[:end])
        pr = m.predict_proba(X[end:v_end])[:, 1]
        ev = evaluator(gs[32000:36000], y[end:v_end], pr)
        th = max(np.arange(0.3, 0.951, 0.01), key=lambda t: (ev(t)["macro_f0.5"], t))
        result = ev(th)
        trials.append({"parameters": conf, "threshold": float(th), "tuning": result})
        p.log(json.dumps(trials[-1]))
        if best is None or result["macro_f0.5"] > best[0]:
            best = (result["macro_f0.5"], m, float(th), conf)
        (BASE / "trials.json").write_text(json.dumps(trials, indent=2))
    _, m, th, conf = best
    # Evaluate fresh holdout once, only after model and threshold selection are complete.
    pr = m.predict_proba(X[v_end:])[:, 1]
    result = evaluator(gs[36000:], y[v_end:], pr)(th)
    report = {
        "training_sample": 40000,
        "fit_entities": 32000,
        "tuning_entities": 4000,
        "holdout_entities": 4000,
        "training_candidate_pairs": len(y),
        "threshold": th,
        "parameters": conf,
        "trials": trials,
        "untouched_holdout": result,
        "holdout_policy": "40,000 new references exclude all earlier 30,000 sampled references and 2,000 development references. Final 4,000 used once after tuning. Delivered model excludes these 4,000.",
    }
    m.booster_.save_model(str(BASE / "evaluation_model.txt"))
    (BASE / "metrics.json").write_text(json.dumps(report, indent=2))
    p.log("HOLDOUT " + json.dumps(result))
    m = LGBMClassifier(**conf)
    m.fit(X[:v_end], y[:v_end])
    m.booster_.save_model(str(BASE / "model.txt"))
    p.log("Final model saved; fresh holdout remains excluded")


def run(root, cache, artifacts, work_dir):
    global ROOT, BASE, CACHE
    ROOT = Path(root)
    CACHE = Path(cache)
    BASE = Path(work_dir)
    BASE.mkdir(parents=True, exist_ok=True)
    p.build(ROOT, "train", CACHE)
    p.build_tokens(CACHE / "train")
    data = select()
    done = 0
    with ProcessPoolExecutor(
        max_workers=8, initializer=init, initargs=(str(CACHE), str(BASE))
    ) as pool:
        for number, n in pool.map(
            batch, [(i // 100, data[i : i + 100]) for i in range(0, len(data), 100)]
        ):
            done += n
            if done % 1000 == 0:
                p.log(f"Features {done}/40000")
    train()
    from calibrate import run as calibrate

    calibrate(CACHE, BASE, BASE / "selected.json", BASE / "final_metrics.json")
    import shutil

    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in ["model.txt", "evaluation_model.txt", "metrics.json", "trials.json"]:
        shutil.copy2(BASE / name, artifacts / name)
    shutil.copy2(BASE / "final_metrics.json", artifacts / "metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--work-dir", type=Path, default=Path("training_work"))
    args = ap.parse_args()
    run(args.data_root, args.cache, args.artifacts, args.work_dir)
