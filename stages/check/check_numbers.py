#!/usr/bin/env python3
"""Compare your run with the published numbers.

The paper's result tables are part of the public dataset (results/*.parquet in the permissive repository). This
script reads them, either from a local download (--dataset DIR) or straight from the Hub (--hf-repo
NatalijaStepurko/orb-mp-embeddings), and compares them with what the stages wrote under your root:

  probe_metrics.parquet           <-> results/{orb,fairchem}/property_prediction/metrics.csv      (every probe score)
  significance_full_metrics.parquet <-> results/proppred_significance/full_metrics.csv           (bootstrap CIs)

For each table it prints the number of rows matched, the largest absolute difference in R² / macro-F1, and
PASS or FAIL against --tol (default 0.005: the scores are deterministic given the seeds, so the differences you
should see are floating-point noise between machines and library builds).

    uv run python stages/check/check_numbers.py --dataset /path/to/orb-mp-embeddings
    uv run python stages/check/check_numbers.py --hf-repo NatalijaStepurko/orb-mp-embeddings
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402


def fetch(args):
    if args.dataset:
        return Path(args.dataset)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(args.hf_repo, repo_type="dataset", allow_patterns=["results/*.parquet"]))


def probe_metrics(ds, tol):
    pub = pd.read_parquet(ds / "results/probe_metrics.parquet")
    pub["score"] = pub.r2.where(pub.task == "reg", pub.macro_f1).astype(float)
    rows = []
    for model, d in (("ORB", P.RESULTS / "orb/property_prediction"), ("UMA", P.RESULTS / "fairchem/property_prediction")):
        f = d / "metrics.csv"
        if not f.exists():
            print(f"  {model}: {f} not found (run stages/probes/probes_{model.lower()}.py first)"); continue
        m = pd.read_csv(f)
        if model == "ORB":
            m["read_out"] = m.rep; m["family"] = None
            # The published ORB "backbone" rows are gnn_stacks_4 (byte-identical tensors);
            # the probe stage writes only gnn_stacks_4, so compare those rows against it.
            if not (m.layer == "backbone").any():
                m = pd.concat([m, m[m.layer == "gnn_stacks_4"].assign(layer="backbone")])
        else:
            m["read_out"] = m.group
        if "estimator" not in m:                  # metrics.csv calls the estimator "model"
            m = m.rename(columns={"model": "estimator"})
        m["model"] = model
        rows.append(m)
    if not rows:
        return None
    mine = pd.concat(rows)
    mine["score"] = mine.r2.where(mine.task == "reg", mine.macro_f1).astype(float)
    key = ["model", "target", "read_out", "family", "layer", "estimator"]
    for c in ("family",):
        pub[c] = pub[c].fillna("-"); mine[c] = mine[c].fillna("-")
    j = pub.merge(mine, on=key, suffixes=("_pub", "_mine"))
    j = j[np.isfinite(j.score_pub) & np.isfinite(j.score_mine)]
    d = (j.score_pub - j.score_mine).abs()
    return len(j), len(pub), float(d.max()) if len(d) else np.nan, j.loc[d.idxmax(), key].to_dict() if len(d) else None


def significance(ds, tol):
    f = P.RESULTS / "proppred_significance/full_metrics.csv"
    if not f.exists():
        print(f"  {f} not found (run stages/probes/significance.py first)"); return None
    pub = pd.read_parquet(ds / "results/significance_full_metrics.parquet")
    mine = pd.read_csv(f)
    key = ["target", "model", "config", "metric"]
    j = pub.merge(mine, on=key, suffixes=("_pub", "_mine"))
    d = (j.point_pub.astype(float) - j.point_mine.astype(float)).abs()
    return len(j), len(pub), float(d.max()) if len(d) else np.nan, j.loc[d.idxmax(), key].to_dict() if len(d) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="local copy of the permissive dataset repository")
    g.add_argument("--hf-repo", help="dataset repository id on the Hugging Face Hub")
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    ds = fetch(a)
    print(f"published tables from {ds}; your results under {P.RESULTS}")
    ok = True
    for name, fn in (("probe scores (metrics.csv vs probe_metrics.parquet)", probe_metrics),
                     ("bootstrap point estimates (full_metrics.csv vs significance_full_metrics.parquet)", significance)):
        r = fn(ds, a.tol)
        if r is None:
            continue
        n, n_pub, dmax, worst = r
        status = "PASS" if (n and dmax <= a.tol) else "FAIL"
        ok &= status == "PASS"
        print(f"  {name}: {n} of {n_pub} published rows matched, max |diff| = {dmax:.4f}  {status}" + (f"  (largest at {worst})" if status == "FAIL" else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
