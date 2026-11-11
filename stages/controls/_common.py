"""Shared argument block and loader for the control entry points (stages/controls/*.py), all of which call into
mlipprobes/significance_lib.py, the one module that holds the control implementations."""
import argparse
from pathlib import Path

from mlipprobes import paths as P
from mlipprobes import significance_lib as L


def parser(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb/property_prediction"), help="ORB-v3 probe outputs (metrics.csv, features/, splits/)")
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem/property_prediction"), help="UMA-S probe outputs")
    ap.add_argument("--controls-output-dir", default=str(P.RESULTS / "proppred_controls"), help="each control writes its own subfolder here")
    ap.add_argument("--targets", default=None, help="comma list; default = the nine reported properties")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    return ap


def load(args):
    orb_dir, uma_dir = Path(args.orb_dir), Path(args.uma_dir)
    orb_rows = L.O.load_metrics(orb_dir / "metrics.csv")
    uma_rows = L.U.load_metrics(uma_dir / "metrics.csv")
    targets = ([L.O.TARGET_ALIASES.get(t.strip(), t.strip()) for t in args.targets.split(",")] if args.targets
               else list(L.RELIABLE_TARGETS))
    ctl = Path(args.controls_output_dir); ctl.mkdir(parents=True, exist_ok=True)
    return orb_dir, uma_dir, orb_rows, uma_rows, targets, ctl
