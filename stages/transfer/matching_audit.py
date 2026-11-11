#!/usr/bin/env python3
"""
stages/transfer/matching_audit.py -- two checks on how the external sets (stages/data/fetch_external.py) relate to MP.

1. Disjoint leakage. 13's StructureMatcher prefilter keeps only MP candidates with the same reduced formula AND the
   same site count, so an equivalent structure stored as a different cell (primitive vs conventional, a supercell)
   can be called "disjoint". A random sample of disjoint entries is re-matched against every MP structure of the
   same reduced formula, any site count, with StructureMatcher(primitive_cell=True, scale=True,
   attempt_supercell=True); the match rate bounds the leakage.
2. Cited ids. An external entry that cites an MP id is "in the MP corpus" on the citation alone. A random sample of
   cited entries is matched geometrically against the cited MP structure (same matcher).
3. OQMD coordinates. 13 builds OQMD structures from the API's `sites` strings as fractional coordinates. For a
   sample of entries the space group recomputed from the built structure (SpacegroupAnalyzer, symprec 0.1) is
   compared with the space-group symbol the API reports (converted to its number).

Writes $MLIP_PROBES_ROOT/results/external_transfer/matching_audit.json

    uv run python stages/transfer/matching_audit.py --n 1000
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import common as qc  # noqa: E402

MP = P.STRUCTURES
EXT = P.STRUCTURES_EXTERNAL
OUT = (P.RESULTS / "external_transfer/matching_audit.json")


def mp_by_formula():
    from pymatgen.core import Composition
    excluded = qc.excluded_materials()
    pre = defaultdict(list)
    for line in open(MP / "index.jsonl"):
        r = json.loads(line)
        if r["mp_id"] in excluded:
            continue
        try:
            pre[Composition(r["formula"]).reduced_formula].append(r["mp_id"])
        except Exception:
            pass
    return pre


def load(p):
    from pymatgen.core import Structure
    return Structure.from_file(str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="sample size per check and database")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.symmetry.groups import SpaceGroup
    rng = np.random.default_rng(args.seed)
    matcher = StructureMatcher(primitive_cell=True, scale=True, attempt_supercell=True)
    pre = mp_by_formula()
    out = {"n_per_check": args.n, "matcher": "StructureMatcher(primitive_cell=True, scale=True, attempt_supercell=True)"}
    for ds in ("jarvis_dft_3d", "oqmd"):
        recs = [json.loads(l) for l in open(EXT / ds / "index.jsonl")]
        dis = [r for r in recs if not r["in_mp_corpus"] and not r.get("structure_match_mp")]
        cited = [r for r in recs if r["in_mp_corpus"]]
        res = {"n_disjoint": len(dis), "n_cited": len(cited)}
        # 1. disjoint leakage, any site count
        t0 = time.time(); hits, examples, n_cand = 0, [], 0
        sample = rng.choice(len(dis), min(args.n, len(dis)), replace=False)
        for i in sample:
            r = dis[i]; s = load(r["cif_file"])
            cands = pre.get(s.composition.reduced_formula, [])
            n_cand += len(cands)
            for m in cands:
                try:
                    if matcher.fit(s, load(MP / f"{m}.cif")):
                        hits += 1; examples.append((r["mp_id"], m, r["nsites"])); break
                except Exception:
                    continue
        res["disjoint_sampled"] = int(len(sample)); res["disjoint_matched_any_cell"] = hits
        res["disjoint_leak_rate"] = hits / len(sample); res["disjoint_examples"] = examples[:10]
        res["mean_candidates_per_entry"] = n_cand / len(sample); res["disjoint_check_seconds"] = round(time.time() - t0)
        print(f"{ds}: disjoint leak {hits}/{len(sample)} ({100 * hits / len(sample):.1f} %), {time.time() - t0:.0f}s", flush=True)
        # 2. cited ids matched geometrically
        if cited:
            t0 = time.time(); fit, no_mp = 0, 0
            sample = rng.choice(len(cited), min(args.n, len(cited)), replace=False)
            for i in sample:
                r = cited[i]; p = MP / f"{r['mpid_reported']}.cif"
                if not p.exists():
                    no_mp += 1; continue
                try:
                    fit += bool(matcher.fit(load(r["cif_file"]), load(p)))
                except Exception:
                    pass
            res["cited_sampled"] = int(len(sample)); res["cited_geometric_match"] = fit; res["cited_match_rate"] = fit / (len(sample) - no_mp)
            print(f"{ds}: cited ids geometrically matched {fit}/{len(sample) - no_mp} ({100 * fit / (len(sample) - no_mp):.1f} %), {time.time() - t0:.0f}s", flush=True)
        # 3. OQMD coordinates: space group of the built structure vs the API's
        if ds == "oqmd":
            pages = sorted((EXT / "_oqmd" / "pages").glob("offset_*.json"))
            api = {}
            for p in pages:
                for e in json.loads(p.read_text()).get("data", []):
                    api[f"oqmd-{e['entry_id']}"] = e.get("spacegroup")
            sample = rng.choice(len(recs), min(args.n, len(recs)), replace=False)
            agree, n_ok, bad = 0, 0, []
            for i in sample:
                r = recs[i]; ref = api.get(r["mp_id"])
                if ref is None:
                    continue
                try:
                    num = SpacegroupAnalyzer(load(r["cif_file"]), symprec=0.1).get_space_group_number()
                except Exception:
                    continue
                n_ok += 1
                try:                                   # the API reports a Hermann-Mauguin symbol, e.g. "P63/mmc"
                    ref_num = SpaceGroup(str(ref)).int_number
                except Exception:
                    ref_num = None
                if ref_num == num:
                    agree += 1
                elif len(bad) < 10:
                    bad.append((r["mp_id"], ref, num))
            res["oqmd_spacegroup_sampled"] = n_ok; res["oqmd_spacegroup_agree"] = agree
            res["oqmd_spacegroup_agree_rate"] = agree / max(n_ok, 1); res["oqmd_spacegroup_disagreements"] = bad
            print(f"oqmd: space group agrees {agree}/{n_ok} ({100 * agree / max(n_ok, 1):.1f} %)", flush=True)
        out[ds] = res
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
