#!/usr/bin/env python3
"""
stages/controls/random_init.py — does pretraining explain decodability, or just architecture?

Compares each target's pretrained best-config XGBoost score against the SAME
architecture/layer/rep but with the model's weights randomly reinitialised instead
of loaded from the pretrained checkpoint (produced by
02_extract_embeddings_{orb,fairchem}.py --random-init, run through the unmodified
05_property_prediction_{orb,fairchem}.py against a separate results tree). The
comparison is held at the SAME config on both sides (not each side's own best) so
the only thing that differs is whether the weights were pretrained -- the causal
question this control exists to answer. Random-init's own best-across-all-configs
score is reported alongside as secondary context (how good can an untrained network
get if you also search over its layers), not as the primary comparison.

Reads  $MLIP_PROBES_ROOT/results/{orb,fairchem}/property_prediction/metrics.csv           (pretrained)
       $MLIP_PROBES_ROOT/results/{orb,fairchem}_random_init/property_prediction/metrics.csv (random-init)
Writes $MLIP_PROBES_ROOT/results/random_init_control/

Usage:
    uv run python stages/controls/random_init.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import csv
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
O = importlib.import_module("stages.probes.probes_orb")
U = importlib.import_module("stages.probes.probes_uma")
S = importlib.import_module("mlipprobes.significance_lib")


def gap_for_target(mod, pretrained_rows, random_rows, target):
    cfg, pretrained_score = mod.best_config(pretrained_rows, target)
    if cfg is None:
        return None
    pm = mod.primary_metric_name(target)
    if mod is O:
        rep, layer = cfg
        cfg_str = f"{rep}/{layer}"
        random_same_cfg = mod._metric_lookup(random_rows, target, rep, layer, "xgb", pm)
    else:
        group, fam, layer = cfg
        cfg_str = f"{group}/{fam}/{layer}"
        # UMA's module names this helper _lookup, not _metric_lookup (ORB's name) --
        # same signature (rows, target, ..., model, metric), different function object.
        random_same_cfg = mod._lookup(random_rows, target, group, fam, layer, "xgb", pm)
    random_best_cfg, random_best_score = mod.best_config(random_rows, target)
    return dict(target=target, config=cfg_str, metric=pm,
               pretrained_score=pretrained_score,
               random_init_same_config=random_same_cfg,
               pretraining_gap=(pretrained_score - random_same_cfg
                                if not np.isnan(random_same_cfg) else float("nan")),
               random_init_best_config=(str(random_best_cfg) if random_best_cfg else None),
               random_init_best_score=(random_best_score if random_best_cfg else float("nan")))


def gap_by_layer(mod, pretrained_rows, random_rows, target):
    """Pretrained and random-init XGBoost scores at every layer of the pretrained best configuration's
    read-out (ORB: its rep; UMA: its group and family): how the pretraining gap changes with depth."""
    cfg, _ = mod.best_config(pretrained_rows, target)
    if cfg is None:
        return []
    pm = mod.primary_metric_name(target)
    if mod is O:
        rep, best_layer = cfg
        look = lambda rows, l: mod._metric_lookup(rows, target, rep, l, "xgb", pm)
        layers, readout = list(O.ALL_LAYERS), rep
    else:
        group, fam, best_layer = cfg
        look = lambda rows, l: mod._lookup(rows, target, group, fam, l, "xgb", pm)
        layers, readout = list(U.GROUP_LAYERS[group]), f"{group}/{fam}"
    out = []
    for i, l in enumerate(layers):
        a, b = look(pretrained_rows, l), look(random_rows, l)
        out.append(dict(target=target, readout=readout, layer=l, depth=i, metric=pm, pretrained_score=a,
                        random_init_score=b, pretraining_gap=a - b, is_selected_layer=(l == best_layer)))
    return out


def same_split(mod, pre_dir: Path, ran_dir: Path, target: str):
    """True when the random-init tree was scored on the pretrained tree's seed-42 split (same train and
    test ids); None when a split file is missing."""
    a, b = (d / "splits" / f"{mod.safe(target)}.npz" for d in (pre_dir, ran_dir))
    if not (a.exists() and b.exists()):
        return None
    a, b = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    return all(set(map(str, a[k])) == set(map(str, b[k])) for k in ("train_mp_ids", "test_mp_ids"))


def trivial_scores(orb_dir: Path, targets) -> dict:
    """{target: score of the trivial predictor on the paper's test split}: the training-set mean for
    regression (R², 0 up to rounding) and the training-set majority class for classification (macro-F1).
    The floor that both the random network and Magpie should be read against."""
    from sklearn.metrics import f1_score, r2_score
    canon, yd = O.load_canonical_targets(orb_dir / "features", list(targets))
    out = {}
    for t in targets:
        sp = np.load(orb_dir / "splits" / f"{O.safe(t)}.npz", allow_pickle=True)
        y = yd[t]; ytr, yte = y[sp["train_idx"]], y[sp["test_idx"]]
        if O.is_clf(t):
            ytr, yte = ytr.astype(str), yte.astype(str)
            cls, cnt = np.unique(ytr, return_counts=True)
            out[t] = float(f1_score(yte, np.full(len(yte), cls[cnt.argmax()]), average="macro"))
        else:
            ytr, yte = ytr.astype(float), yte.astype(float)
            ytr, yte = ytr[np.isfinite(ytr)], yte[np.isfinite(yte)]
            out[t] = float(r2_score(yte, np.full(len(yte), ytr.mean())))
    return out


def magpie_scores(path: Path) -> dict:
    """{target: Magpie XGBoost test score} from the composition baseline (06 --composition-baseline):
    the untrained-featuriser reference on the same split."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["target"]: float(r["comp_xgb"]) for r in csv.DictReader(f) if r.get("features") == "magpie"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb/property_prediction"))
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem/property_prediction"))
    ap.add_argument("--orb-random-dir", default=str(P.RESULTS / "orb_random_init/property_prediction"))
    ap.add_argument("--uma-random-dir", default=str(P.RESULTS / "fairchem_random_init/property_prediction"))
    ap.add_argument("--output-dir", default=str(P.RESULTS / "random_init_control"))
    ap.add_argument("--magpie-csv", default=str(P.RESULTS / "proppred_controls/composition_baseline/composition_baseline.csv"))
    ap.add_argument("--targets", default=None, help="comma list; default = all 9 reliable")
    args = ap.parse_args()

    orb_dir, uma_dir = Path(args.orb_dir), Path(args.uma_dir)
    orb_r_dir, uma_r_dir = Path(args.orb_random_dir), Path(args.uma_random_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.targets.split(",") if args.targets else list(S.RELIABLE_TARGETS)

    orb_rows = O.load_metrics(orb_dir / "metrics.csv")
    uma_rows = U.load_metrics(uma_dir / "metrics.csv")
    orb_r_rows = O.load_metrics(orb_r_dir / "metrics.csv")
    uma_r_rows = U.load_metrics(uma_r_dir / "metrics.csv")

    magpie = magpie_scores(Path(args.magpie_csv))
    trivial = trivial_scores(orb_dir, targets)
    records = []
    for t in targets:
        for name, mod, pre, ran, pdir, rdir in (("orb", O, orb_rows, orb_r_rows, orb_dir, orb_r_dir),
                                                ("uma", U, uma_rows, uma_r_rows, uma_dir, uma_r_dir)):
            rec = gap_for_target(mod, pre, ran, t)
            if rec is None:
                print(f"  [skip] {t} ({name}): no best config")
                continue
            rec["model"] = name
            rec["magpie_score"] = magpie.get(t, float("nan"))
            rec["trivial_score"] = trivial.get(t, float("nan"))
            rec["random_minus_magpie"] = rec["random_init_same_config"] - rec["magpie_score"]
            rec["random_best_minus_magpie"] = rec["random_init_best_score"] - rec["magpie_score"]
            rec["same_split"] = same_split(mod, pdir, rdir, t)
            if rec["same_split"] is False:
                print(f"  [WARNING] {t} ({name}): the random-init tree used a different split from the pretrained tree")
            records.append(rec)
            print(f"  {t:24s} {name:3s} cfg={rec['config']:18s} "
                 f"pretrained={rec['pretrained_score']:.3f} "
                 f"random_init(same cfg)={rec['random_init_same_config']:.3f} "
                 f"gap={rec['pretraining_gap']:+.3f}")

    with open(out_dir / "pretraining_gap_table.csv", "w", newline="") as f:
        fields = ["target", "model", "config", "metric", "pretrained_score",
                  "random_init_same_config", "pretraining_gap",
                  "random_init_best_config", "random_init_best_score",
                  "magpie_score", "trivial_score", "random_minus_magpie", "random_best_minus_magpie", "same_split"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(records)

    # the gap at every layer of the selected read-out
    layer_rows = []
    for t in targets:
        for name, mod, pre, ran in (("orb", O, orb_rows, orb_r_rows), ("uma", U, uma_rows, uma_r_rows)):
            for r in gap_by_layer(mod, pre, ran, t):
                r["model"] = name; r["magpie_score"] = magpie.get(t, float("nan")); layer_rows.append(r)
    if layer_rows:
        with open(out_dir / "pretraining_gap_by_layer.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["target", "model", "readout", "layer", "depth", "metric", "pretrained_score",
                                              "random_init_score", "pretraining_gap", "is_selected_layer", "magpie_score"])
            w.writeheader(); w.writerows(layer_rows)
        fig, axes = plt.subplots(2, len(targets), figsize=(2.6 * len(targets), 5.2), sharey=True, squeeze=False)
        for i, name in enumerate(("orb", "uma")):
            for j, t in enumerate(targets):
                ax = axes[i][j]
                rr = [r for r in layer_rows if r["model"] == name and r["target"] == t]
                if not rr:
                    ax.axis("off"); continue
                x = [r["depth"] for r in rr]
                col = "#0072B2" if name == "orb" else "#D55E00"
                ax.plot(x, [r["pretrained_score"] for r in rr], color=col, marker="o", ms=3.5, lw=1.8, label="pretrained")
                ax.plot(x, [r["random_init_score"] for r in rr], color=col, marker="o", ms=3.5, lw=1.8, ls="--", alpha=0.6,
                        mfc="white", label="random weights")
                if not np.isnan(rr[0]["magpie_score"]):
                    ax.axhline(rr[0]["magpie_score"], color="#1f1f1e", lw=1, ls=":", label="Magpie")
                lo = np.nanmin([r["random_init_score"] for r in rr] + [r["pretrained_score"] for r in rr])
                ax.set_ylim(max(min(lo, 0.0), -0.5) - 0.05, 1.02)
                ax.set_title(f"{t}\n{name.upper()} {rr[0]['readout']}", fontsize=7.5)
                ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x], fontsize=7)
                if i == 1:
                    ax.set_xlabel("layer (depth)", fontsize=8)
            axes[i][0].set_ylabel("R2 / macro-F1", fontsize=8)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out_dir / "pretraining_gap_by_layer.png", dpi=140); plt.close(fig)

    # the gap itself against depth: one panel per target, both models (supplementary figure)
    if layer_rows:
        order = [t for t in ["formation_energy/atom", "energy_above_hull", "band_gap", "density", "bulk_modulus",
                             "mag_density", "crystal_system", "metal_insulator", "magnetic_label"] if t in targets]
        order += [t for t in targets if t not in order]
        ncol = 3; nrow = int(np.ceil(len(order) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), sharex=True, squeeze=False)
        col = {"orb": "#0072B2", "uma": "#D55E00"}; lab = {"orb": "ORB-v3", "uma": "UMA-S"}
        for ax, t in zip(axes.flat, order):
            ax.axhline(0, color="#6b6a64", lw=0.8, ls=":")
            sub = []
            for name in ("orb", "uma"):
                rr = [r for r in layer_rows if r["model"] == name and r["target"] == t]
                if not rr:
                    continue
                x = [r["depth"] for r in rr]; g = [r["pretraining_gap"] for r in rr]
                ax.plot(x, g, color=col[name], marker="o", ms=3.5, lw=1.8, label=lab[name])
                sel = [r for r in rr if r["is_selected_layer"]]
                if sel:
                    ax.plot([sel[0]["depth"]], [sel[0]["pretraining_gap"]], marker="o", ms=9, mfc="none", mec=col[name], mew=1.6, ls="none")
                sub.append(f"{lab[name]} {rr[0]['readout']}")
            ax.set_title(f"{t}\n" + " · ".join(sub), fontsize=8)
            ax.set_xticks(range(6)); ax.tick_params(labelsize=8)
        for ax in axes.flat[len(order):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("layer (0 = encoder / embed)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("pretrained − random-init", fontsize=8)
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.975))
        fig.suptitle("Pretrained minus random-init score at every layer of the selected read-out (circle = selected layer)", fontsize=9, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_dir / "pretraining_gap_vs_depth.png", dpi=150); plt.close(fig)

    # markdown table
    lines = ["# Random-init control: does pretraining explain decodability?", "",
             "Same architecture/layer/rep on both sides; only the weights differ "
             "(pretrained checkpoint vs a random reinitialisation, "
             "stages/extract/extract_{orb,uma}.py --random-init). `gap` = pretrained - random-init, "
             "at the SAME config. `random-init best` searches over all of random-init's own "
             "configs too (secondary context, not the primary comparison). `Magpie` = composition-only "
             "XGBoost on the same split: the reference that needs no network at all. `Trivial` = the training "
             "mean (regression, R²) or the majority class (classification, macro-F1) on the test split.", "",
             "| Target | Model | Config | Metric | Pretrained | Random-init (same cfg) | Gap | Random-init best (any cfg) | Magpie | Trivial |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['target']} | {r['model']} | {r['config']} | {r['metric']} | "
                    f"{r['pretrained_score']:.3f} | {r['random_init_same_config']:.3f} | "
                    f"{r['pretraining_gap']:+.3f} | {r['random_init_best_score']:.3f} "
                    f"({r['random_init_best_config']}) | {r['magpie_score']:.3f} | {r['trivial_score']:.3f} |")
    (out_dir / "pretraining_gap_table.md").write_text("\n".join(lines) + "\n")

    # plot: paired bars, pretrained vs random-init(same cfg), per target/model
    fig, ax = plt.subplots(figsize=(9, 5))
    targets_plotted = sorted({r["target"] for r in records})
    x = np.arange(len(targets_plotted))
    width = 0.2
    offsets = {("orb", "pretrained"): -1.5, ("orb", "random"): -0.5,
              ("uma", "pretrained"): 0.5, ("uma", "random"): 1.5}
    colors = {"orb_pretrained": "#0072B2", "orb_random": "#A6CEE3",
             "uma_pretrained": "#D55E00", "uma_random": "#FDBF6F"}
    for name in ("orb", "uma"):
        pre_vals = [next((r["pretrained_score"] for r in records
                          if r["target"] == t and r["model"] == name), np.nan) for t in targets_plotted]
        rand_vals = [next((r["random_init_same_config"] for r in records
                           if r["target"] == t and r["model"] == name), np.nan) for t in targets_plotted]
        ax.bar(x + offsets[(name, "pretrained")] * width, pre_vals, width,
              label=f"{name.upper()} pretrained", color=colors[f"{name}_pretrained"])
        ax.bar(x + offsets[(name, "random")] * width, rand_vals, width,
              label=f"{name.upper()} random-init", color=colors[f"{name}_random"])
    mg = [magpie.get(t, np.nan) for t in targets_plotted]
    ax.scatter(x, mg, marker="_", s=900, color="#1f1f1e", linewidths=1.6, zorder=5, label="Magpie (composition only)")
    ax.set_xticks(x); ax.set_xticklabels(targets_plotted, rotation=30, ha="right")
    ax.set_ylabel("R2 / macro-F1 (primary metric)")
    ax.set_title("Pretrained vs random-init, same config")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "pretraining_gap.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
