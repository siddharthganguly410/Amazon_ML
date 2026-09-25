import json, sys, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from lightgbm import Booster
import pipeline as p


def init(cache, art):
    global IDX, EVAL, FINAL
    IDX = p.Index(Path(cache) / "train")
    EVAL = Booster(model_file=str(Path(art) / "evaluation_model.txt"))
    FINAL = Booster(model_file=str(Path(art) / "model.txt"))


def batch(items):
    xx = []
    meta = []
    for r, truth in items:
        cs = IDX.candidates(r)
        start = len(xx)
        xx.extend(p.features(r, b) for b in cs)
        meta.append((r, truth, [b[0] for b in cs], start, len(xx)))
    X = np.asarray(xx, dtype=np.float32)
    pe = EVAL.predict(X, num_threads=1) if len(X) else []
    pf = FINAL.predict(X, num_threads=1) if len(X) else []
    return [
        (r, truth, ids, pe[s:e].tolist(), pf[s:e].tolist())
        for r, truth, ids, s, e in meta
    ]


def metric(rows, th, which):
    scores = []
    hit = truth_total = pred_total = blocked = 0
    for r, gt, ids, pe, pf in rows:
        gt = set(gt)
        probs = pe if which == "evaluation" else pf
        pred = {v for v, prob in zip(ids, probs) if prob >= th}
        scores.append(p.score(gt, pred))
        hit += len(gt & pred)
        truth_total += len(gt)
        pred_total += len(pred)
        blocked += len(gt & set(ids))
    return {
        "macro_f0.5": float(np.mean(scores)),
        "pair_precision": hit / max(1, pred_total),
        "pair_recall": hit / max(1, truth_total),
        "blocking_recall": blocked / max(1, truth_total),
        "entities": len(rows),
    }


def run(cache, art, selected, result_path):
    selected = json.loads(Path(selected).read_text())[32000:]
    results = []
    with ProcessPoolExecutor(
        max_workers=8, initializer=init, initargs=(str(cache), str(art))
    ) as pool:
        for result in pool.map(
            batch, [selected[i : i + 100] for i in range(0, len(selected), 100)]
        ):
            results.extend(result)
            if len(results) % 1000 == 0:
                p.log(f"Inference-policy calibration {len(results)}/8000")
    # Threshold uses only the original 4,000 tuning entities and the 32,000-fit model.
    tune = results[:4000]
    hold = results[4000:]
    th = max(
        np.arange(0.3, 0.951, 0.01),
        key=lambda t: (metric(tune, t, "evaluation")["macro_f0.5"], t),
    )
    report = json.loads((Path(art) / "metrics.json").read_text())
    report["broad_policy_pre_refit_holdout"] = report.pop("untouched_holdout", None)
    report["threshold"] = float(th)
    report["inference_posting_limit"] = 2000
    report["training_posting_limit"] = 5000
    report["tuning"] = metric(tune, th, "evaluation")
    report["untouched_holdout"] = metric(hold, th, "final")
    report["country_holdout"] = {
        c: metric([row for row in hold if row[0][3] == c], th, "final")
        for c in sorted({row[0][3] for row in hold})
    }
    report["holdout_policy"] = (
        "Final 4,000 entities excluded from all fitting and threshold selection. Score evaluates the delivered model trained on 36,000 entities, using the 2,000-posting inference policy chosen on development data. All 40,000 exclude the earlier 30,000 sample and 2,000 development entities."
    )
    Path(result_path).write_text(json.dumps(report, indent=2))
    p.log(json.dumps(report["untouched_holdout"]))
