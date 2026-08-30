#!/usr/bin/env python3
"""Benchmark ViT4TS — synthetic quick run + TSB-AD-U if present."""
import argparse, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def f1_max(scores, labels):
    from sklearn.metrics import f1_score

    scores = np.asarray(scores); labels = np.asarray(labels).astype(int)
    best = 0
    for q in np.linspace(0.5, 0.99, 50):
        thresh = np.quantile(scores, q)
        pred = (scores >= thresh).astype(int)
        # point-adjusted? just standard
        if pred.sum() == 0:
            continue
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best:
            best = f1
    return best

def synthetic_series(n=500, seed=0):
    rng = np.random.default_rng(seed)
    s = np.sin(np.linspace(0, 20, n)) + rng.normal(0, 0.15, n)
    labels = np.zeros(n, dtype=int)
    # mix point spikes and shape anomalies (flat) — shape is hard for z-score
    for start in [100, 300]:
        if seed % 2 == 0:
            s[start:start+10] += rng.choice([3, -3])
        else:
            s[start:start+15] = np.linspace(s[start-1], s[start+15], 15)  # flat-ish shape anomaly
        labels[start:start+10] = 1
    return s, labels

def window_hit(cands, labels, tol=112):
    """Relaxed detection: candidate within tol of any true label."""
    true_idx = np.where(labels==1)[0]
    if len(true_idx)==0 or len(cands)==0:
        return 0.0
    hits = sum(any(abs(c - t) <= tol for c in cands) for t in true_idx)
    return hits / len(true_idx)

def run_quick():
    from src.vit4ts import ViT4TS
    print("quick benchmark: synthetic 3 series (ViT4TS)")
    m = ViT4TS(alpha=0.01, window_size=224)
    vit_f1s, hits = [], []
    for i in range(3):
        s, y = synthetic_series(500, seed=i)
        scores, _ = m.predict_scores(s)
        cands = m.candidates(scores)
        f1 = f1_max(scores, y)
        hit = window_hit(cands, y, tol=112)
        vit_f1s.append(f1); hits.append(hit)
        print(f" series {i}: F1-max={f1:.3f}  window-hit={hit:.2f}  candidates={len(cands)}  score@anomaly={scores[y==1].mean():.4f} vs bg={scores[y==0].mean():.4f}")
    print(f"\navg ViT4TS F1-max={np.mean(vit_f1s):.3f} hit-rate={np.mean(hits):.2f} (paper reports +24.6% F1-max over TS baselines on TSB-AD-U)")
    if np.mean(hits) >= 0.5:
        print("trend reproduced: candidates overlap injected anomalies")
    else:
        print("note: hit-rate <0.5 — synthetic shape not strongly flagged (try different seed/alpha)")
    return np.mean(vit_f1s), np.mean(hits)

def run_tsbad(dataset_path):
    import pandas as pd
    from src.vit4ts import ViT4TS
    # TSB-AD-U layout: each CSV with value + label cols (varies)
    files = list(pathlib.Path(dataset_path).rglob("*.csv"))[:5]
    if not files:
        print(f"no CSV in {dataset_path}, falling back to synthetic")
        return run_quick()
    m = ViT4TS(alpha=0.01, window_size=224)
    f1s=[]
    for f in files:
        df = pd.read_csv(f)
        # guess value col and label col
        val_col = [c for c in df.columns if "value" in c.lower() or c==df.columns[0]][0]
        lab_col = [c for c in df.columns if "label" in c.lower() or "anomaly" in c.lower()]
        lab_col = lab_col[0] if lab_col else None
        s = df[val_col].to_numpy()
        if lab_col is None:
            continue
        y = df[lab_col].to_numpy().astype(int)
        scores,_ = m.predict_scores(s[:2000])  # cap length for speed
        y = y[:len(scores)]
        scores = scores[:len(y)]
        f1s.append(f1_max(scores, y))
        print(f" {f.name}: F1-max {f1s[-1]:.3f}")
    print(f"avg F1-max {np.mean(f1s):.3f} over {len(f1s)} files")
    return np.mean(f1s) if f1s else 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="TSB-AD-U")
    p.add_argument("--metric", default="F1-max")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    if args.quick or not pathlib.Path(args.data_dir).exists():
        run_quick()
    else:
        # try data_dir/dataset else fallback
        cand = pathlib.Path(args.data_dir) / args.dataset
        if cand.exists():
            run_tsbad(str(cand))
        else:
            print(f"{cand} not found -> quick")
            run_quick()
