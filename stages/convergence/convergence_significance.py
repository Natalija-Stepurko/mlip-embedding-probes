"""
stages/convergence/convergence_significance.py — robustness of the ORB↔UMA convergence result.

convergence.py reports that ORB
and UMA converge above chance on aligned atoms. This stage quantifies how robust the peak
convergence is by resampling atoms many times and comparing the peak cross-model CKA
against its permutation baseline over the repeats, reporting means, 95% confidence
intervals, the stability of the peak layer-pair, and an empirical p-value (fraction of
repeats where the real peak did not exceed the permuted one). Mutual k-NN is resampled
alongside it, at the layer pair convergence.py reports as its peak.

Read-out and residualisation follow convergence.py's reported grids: the ang_norm channel, with
element identity residualised out of both models (--no-residualise disables this, as a
diagnostic only). A control on a different footing from the grids it certifies would not
certify them -- the earlier full/l0 unresidualised runs peaked at the near-input layers,
where the agreement is the shared input rather than a shared representation.

For efficiency a large atom *pool* is collected from disk once; each repeat then draws a
fresh random subset (its own atoms and its own permutation) from that pool — the atom-level
analog of the bio script's per-repeat residue resampling, without re-reading tens of
thousands of .pt files per repeat.

Outputs (under --out-dir, default results/convergence_orb_uma/significance), per UMA variant:
    summary.txt           peak/baseline means, 95% CIs, modal peak layer-pair, p-value
    significance.png       real peak vs permutation baseline with 95% CIs (per variant)

Usage:
    uv run python scripts/stages/convergence/convergence_significance.py \\
        --orb-dir $MLIP_PROBES_ROOT/results/orb --uma-dir $MLIP_PROBES_ROOT/results/fairchem \\
        --out-dir $MLIP_PROBES_ROOT/results/convergence_orb_uma/significance --repeats 25 \\
        --uma-feature ang_norm
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import common as qc
C = importlib.import_module("stages.convergence.convergence")  # reuse collection + helpers


def _peak_cka_once(orb: dict, uma: dict, rng, els=None) -> tuple[float, tuple[int, int], float]:
    """One resample: optionally residualise element identity, then center, compute the
    CKA grid + permuted grid, return peak/loc/perm-peak.

    When `els` is given, the per-element mean is removed from BOTH models first, exactly
    as in convergence.py's *_resid grids. Both models are given atomic numbers, so agreement
    that only reflects the periodic table is not evidence of learned convergence; the
    reported grids (Fig. S8, Table S15) are residualised, so this control must be too.
    """
    if els is not None:
        orb = {L: C.residualise_elements(orb[L], els) for L in C.ORB_LAYERS}
        uma = {L: C.residualise_elements(uma[L], els) for L in C.UMA_LAYERS}
    orb_c = {L: C._center(orb[L]) for L in C.ORB_LAYERS}
    uma_c = {L: C._center(uma[L]) for L in C.UMA_LAYERS}
    perm = rng.permutation(next(iter(orb.values())).shape[0])
    nE, nS = len(C.ORB_LAYERS), len(C.UMA_LAYERS)
    cka = np.empty((nE, nS)); ckap = np.empty((nE, nS))
    for i, LO in enumerate(C.ORB_LAYERS):
        for j, LU in enumerate(C.UMA_LAYERS):
            cka[i, j] = qc.linear_cka(orb_c[LO], uma_c[LU])
            ckap[i, j] = qc.linear_cka(orb_c[LO], uma_c[LU][perm])
    ij = np.unravel_index(np.nanargmax(cka), cka.shape)
    return float(np.nanmax(cka)), (int(ij[0]), int(ij[1])), float(np.nanmax(ckap))


def _ci(x):
    x = np.asarray(x)
    return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def _mknn_peak_pair(grid_key):
    """Stage-08 peak (ORB, UMA) layer indices of the mutual k-NN grid.

    Restored from commit 735d097; `grid_key` selects the same residualisation footing
    the resampling uses, so the pair evaluated here is the pair convergence.py reports.
    """
    g = np.load((P.RESULTS / "convergence_orb_uma/grids.npz"), allow_pickle=True)
    grid = g[f"{grid_key}__mutual_knn"]
    ij = np.unravel_index(np.nanargmax(grid), grid.shape)
    return int(ij[0]), int(ij[1])


def _run_variant(orb_pool, uma_pool, variant, repeats, sub, seed, out_dir, els_pool=None):
    uma_raw_variant = C._uma_variant(uma_pool, variant)
    Npool = next(iter(orb_pool.values())).shape[0]
    sub = min(sub, Npool)
    grid_key = f"{variant}_resid" if els_pool is not None else variant
    mi, mj = _mknn_peak_pair(grid_key)
    peaks, peaks_perm, locs = [], [], []
    mkns, mkns_perm = [], []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        sel = rng.choice(Npool, sub, replace=False)
        orb_s = {L: orb_pool[L][sel] for L in C.ORB_LAYERS}
        uma_s = {L: uma_raw_variant[L][sel] for L in C.UMA_LAYERS}
        els_s = els_pool[sel] if els_pool is not None else None
        pk, ij, pkp = _peak_cka_once(orb_s, uma_s, rng, els_s)
        if els_s is not None:
            X = C.residualise_elements(orb_s[C.ORB_LAYERS[mi]], els_s)
            Y = C.residualise_elements(uma_s[C.UMA_LAYERS[mj]], els_s)
        else:
            X, Y = orb_s[C.ORB_LAYERS[mi]], uma_s[C.UMA_LAYERS[mj]]
        mk = C.mutual_knn(X, Y, seed=seed + r)
        mkp = C.mutual_knn(X, Y[rng.permutation(sub)], seed=seed + r)
        peaks.append(pk); peaks_perm.append(pkp); locs.append(ij)
        mkns.append(mk); mkns_perm.append(mkp)
        print(f"  [{variant}] repeat {r+1}/{repeats}: peak CKA={pk:.3f} "
              f"at ORB {C.ORB_LBL[ij[0]]}×UMA {C.UMA_LBL[ij[1]]}  perm={pkp:.3f}", flush=True)

    pm, plo, phi = _ci(peaks)
    qm, qlo, qhi = _ci(peaks_perm)
    mm, mlo, mhi = _ci(mkns)
    nm, nlo, nhi = _ci(mkns_perm)
    modal, modal_n = Counter(locs).most_common(1)[0]
    pval = float(np.mean(np.asarray(peaks) <= np.asarray(peaks_perm)))
    mk_pval = float(np.mean(np.asarray(mkns) <= np.asarray(mkns_perm)))
    lines = [
        f"[UMA node variant: {variant}{'_resid' if els_pool is not None else ''}]  "
        f"{repeats} atom resamples of {sub:,} atoms "
        f"(pool {Npool:,})",
        f"  peak CKA:        mean {pm:.3f}  95% CI [{plo:.3f}, {phi:.3f}]",
        f"  permuted CKA:    mean {qm:.3f}  95% CI [{qlo:.3f}, {qhi:.3f}]",
        f"  separation:      {pm - qm:.3f}",
        f"  modal peak pair: ORB {C.ORB_LBL[modal[0]]} × UMA {C.UMA_LBL[modal[1]]}  "
        f"({modal_n}/{repeats} repeats)",
        f"  empirical p (peak <= permuted): {pval:.3g}",
        f"  mutual k-NN at stage-08 peak pair ORB {C.ORB_LBL[mi]} × UMA {C.UMA_LBL[mj]}:",
        f"    real:     mean {mm:.3f}  95% CI [{mlo:.3f}, {mhi:.3f}]",
        f"    permuted: mean {nm:.4f}  95% CI [{nlo:.4f}, {nhi:.4f}]"
        f"   (analytic chance k/n = {10/(min(4000, sub) - 1):.4f})",
        f"    empirical p (real <= permuted): {mk_pval:.3g}",
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([0, 1], [pm, qm], yerr=[[pm - plo, qm - qlo], [phi - pm, qhi - qm]],
           color=["goldenrod", "lightgray"], capsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real peak CKA", "permutation baseline"])
    ax.set_ylabel("peak cross-model CKA")
    ax.set_title(f"ORB↔UMA convergence robustness  (UMA={variant}; "
                 f"{repeats} resamples; p={pval:.2g})")
    fig.tight_layout()
    fig.savefig(out_dir / f"significance_uma-{variant}.png", dpi=140)
    plt.close(fig)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="ORB↔UMA convergence significance")
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb"))
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem"))
    ap.add_argument("--out-dir", default=str(P.RESULTS / "convergence_orb_uma/significance"))
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--pool-atoms", type=int, default=45000, help="atoms collected once")
    ap.add_argument("--sub-atoms", type=int, default=15000, help="atoms per resample")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--max-mats", type=int, default=30000)
    ap.add_argument("--uma-feature", choices=["ang_norm", "full", "l0", "both"], default="ang_norm")
    ap.add_argument("--no-residualise", action="store_true",
                    help="skip element residualisation; the reported grids (Fig. S8, "
                         "Table S15) ARE residualised, so this is a diagnostic only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["full", "l0"] if args.uma_feature == "both" else [args.uma_feature]

    print(f"Collecting atom pool ({args.pool_atoms:,} target) ...")
    orb_pool, uma_pool, els_pool, N = C.collect_aligned(Path(args.orb_dir), Path(args.uma_dir),
                                              args.pool_atoms, args.cap, args.max_mats,
                                              args.seed)
    header = [f"ORB↔UMA convergence significance  (pool N={N:,})", ""]
    out = list(header)
    for variant in variants:
        out += _run_variant(orb_pool, uma_pool, variant, args.repeats,
                            args.sub_atoms, args.seed, out_dir,
                            els_pool=None if args.no_residualise else els_pool) + [""]

    (out_dir / "summary.txt").write_text("\n".join(out).rstrip() + "\n")
    print("\n" + "\n".join(out).rstrip())
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
