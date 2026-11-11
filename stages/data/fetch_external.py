#!/usr/bin/env python3
"""
stages/data/fetch_external.py — fetch an external DFT database and write it in fetch_mp.py's
manifest contract, with every structure's relation to the Materials Project corpus
recorded, so extraction (stages/extract) and transfer scoring (stages/transfer/transfer.py) run on it unchanged.

Why this stage exists. Every number in the paper is trained and tested on Materials
Project structures with Materials Project labels: transfer across *properties*, never
across *data distributions*. An external database tests the latter -- but only if the
structures it contributes are genuinely outside the corpus the probes were fitted on,
and only if its labels are scored in its own units (a different functional shifts
band gaps and re-references formation energies). So this stage does three things and
refuses to blur them:

  1. Download and map targets onto the paper's nine, naming each one that the source
     does not carry (all nine are always reported, present or not).
  2. Write CIFs + index.jsonl exactly as stages/data/fetch_mp.py does (same keys, same
     plausibility bounds), so the rest of the pipeline needs no branches.
  3. Record, per structure, how it relates to the MP corpus:
        in_mp_corpus        the source cites an MP id that is in our 154,875
        structure_match_mp  no cited id, but StructureMatcher finds an MP twin
     transfer.py scores the disjoint slice (neither flag) as *transfer*, and the shared
     slice as *label provenance* (same crystals, different DFT). Pooling the two would
     let leakage masquerade as transfer.

Sources
  jarvis_dft_3d/matminer      matminer's bundled subset (25,923; every row cites an MP id;
                              e_form, OPT/MBJ gap, bulk modulus; no E-hull, no magnetism)
  jarvis_dft_3d/jarvis-tools  the full JARVIS-DFT 3D release via jarvis.db.figshare
                              (adds ehull, magmom, and the JARVIS-only structures)
  oqmd                        a stride sample of the OQMD REST API (PBE; ~1.4M entries with
                              natoms <= 40): formation energy, stability (hull distance),
                              band gap, structure. No bulk modulus, no magnetic moments.
                              Pages are cached under _oqmd/pages/ so the run is resumable.
                              OQMD is the PBE control: JARVIS changes the functional, OQMD
                              keeps PBE but changes the database.

Label caveat recorded in meta.json: JARVIS reports OptB88vdW gaps for metals as small
positive numbers rather than MP's exact 0.0, so mlipprobes.common's metal/insulator rule
(band_gap == 0) does not transfer as-is. The raw gap is written, and transfer.py applies the
same band_gap == 0 rule as for MP.

Writes  <out-dir>/<id>.cif, <out-dir>/index.jsonl, <out-dir>/meta.json
Usage
    uv run python stages/data/fetch_external.py --source matminer
    uv run python stages/data/fetch_external.py --source jarvis-tools
    uv run python stages/data/fetch_external.py --source matminer --no-match --limit 500
    uv run python stages/data/fetch_external.py --source oqmd --oqmd-n 40000
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
F = importlib.import_module("stages.data.fetch_mp")                      # PLAUSIBILITY_BOUNDS, same keys
from mlipprobes import common as qc

MP_STRUCTURES = P.STRUCTURES
DEFAULT_OUT = (P.STRUCTURES_EXTERNAL / "jarvis_dft_3d")
CACHE_HOME = P.STRUCTURES_EXTERNAL           # matminer / jarvis-tools downloads

# paper target  ->  (matminer column, jarvis-tools key)
TARGET_MAP = {
    "formation_energy_per_atom": ("e_form",       "formation_energy_peratom"),
    "band_gap":                  ("gap opt",      "optb88vdw_bandgap"),
    "bulk_modulus":              ("bulk modulus", "bulk_modulus_kv"),
    "energy_above_hull":         (None,           "ehull"),
    "total_magnetization":       (None,           "magmom_oszicar"),
    # density, volume, nsites, crystal_system, space_group: computed from the structure
}
# the nine paper targets and how each is obtained here (for meta.json + the log)
NINE = {
    "formation_energy_per_atom": "source label",
    "energy_above_hull":         "source label (jarvis-tools only)",
    "band_gap":                  "source label, OptB88vdW",
    "density":                   "from structure",
    "bulk_modulus":              "Voigt-Reuss-Hill average K_VRH recomputed from the source elastic tensor (jarvis-tools); the "
                                 "source field bulk_modulus_kv is the Voigt bound K_V, which MP does not report",
    "mag_density":               "total_magnetization / volume (jarvis-tools only)",
    "crystal_system":            "from structure (SpacegroupAnalyzer, symprec 0.1)",
    "metal_insulator":           "from band_gap -- see metal-gap caveat",
    "magnetic_label":            "from total_magnetization / volume (jarvis-tools only)",
}


def _na(v):
    return v is None or v == "na" or (isinstance(v, float) and not np.isfinite(v))


def _num(v):
    if _na(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bounded(key, v):
    """Apply fetch_mp.py's plausibility bounds the way it does: out-of-range -> None."""
    v = _num(v)
    if v is None:
        return None
    lo, hi = F.PLAUSIBILITY_BOUNDS.get(key, (None, None))
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return v


# ----------------------------------------------------------------------------- sources
def k_vrh(elastic_tensor):
    """Voigt-Reuss-Hill bulk modulus (GPa) from a 6x6 Voigt-notation elastic tensor, the quantity MP reports
    as bulk_modulus (K_VRH); None when the tensor is missing, singular or gives a non-physical value."""
    import numpy as _np
    from pymatgen.core.elasticity import ElasticTensor
    if not isinstance(elastic_tensor, list):
        return None
    try:
        with _np.errstate(all="ignore"):
            k = float(ElasticTensor.from_voigt(_np.asarray(elastic_tensor, dtype=float)).k_vrh)
    except Exception:
        return None
    return k if _np.isfinite(k) and k > 0 else None


def load_matminer():
    from matminer.datasets import load_dataset
    df = load_dataset("jarvis_dft_3d", data_home=str(CACHE_HOME / "_matminer"))
    rows = []
    # to_dict, not itertuples()._asdict(): pandas renames tuple fields whose names contain spaces
    # ("gap opt", "bulk modulus") to positional names, so _asdict() lookups of those columns return None
    for d in df.to_dict("records"):
        rows.append(dict(
            id=str(d["jid"]), mpid=(str(d["mpid"]) if not _na(d.get("mpid")) else None),
            structure=d["structure"],
            formation_energy_per_atom=d.get("e_form"), band_gap=d.get("gap opt"),
            band_gap_mbj=d.get("gap tbmbj"), bulk_modulus=k_vrh(d.get("elastic tensor")),
            energy_above_hull=None, total_magnetization=None))
    return rows, dict(source="matminer:jarvis_dft_3d", n=len(rows))


def load_jarvis_tools():
    from jarvis.db.figshare import data
    from jarvis.core.atoms import Atoms
    import os
    cwd = os.getcwd(); (CACHE_HOME / "_jarvis").mkdir(parents=True, exist_ok=True); os.chdir(CACHE_HOME / "_jarvis")     # jarvis caches in cwd
    try:
        d = data("dft_3d")
    finally:
        os.chdir(cwd)
    rows = []
    for e in d:
        ref = e.get("reference")
        mpid = str(ref) if (not _na(ref) and str(ref).startswith(("mp-", "mvc-"))) else None
        try:
            structure = Atoms.from_dict(e["atoms"]).pymatgen_converter()
        except Exception:
            continue
        rows.append(dict(
            id=str(e["jid"]), mpid=mpid, structure=structure,
            formation_energy_per_atom=e.get("formation_energy_peratom"),
            band_gap=e.get("optb88vdw_bandgap"), band_gap_mbj=e.get("mbj_bandgap"),
            bulk_modulus=k_vrh(e.get("elastic_tensor")), energy_above_hull=e.get("ehull"),
            total_magnetization=e.get("magmom_oszicar")))
    return rows, dict(source="jarvis-tools:dft_3d", n=len(rows))


OQMD_API = "https://oqmd.org/oqmdapi/formationenergy"
OQMD_FIELDS = "entry_id,name,spacegroup,natoms,band_gap,delta_e,stability,unit_cell,sites,volume"
OQMD_FILTER = "natoms%3C%3D40"          # natoms<=40, URL-encoded (a bare '<' is a 400)
OQMD_PAGE = 100


def load_oqmd(n_target: int = 40000, page: int = OQMD_PAGE):
    """A stride sample of OQMD: n_target/page pages spread evenly over the whole database
    (offsets i*stride), so the sample is not the first N entries. Each page is cached as
    JSON; a rerun re-reads the cache and only fetches what is missing."""
    import json as _json, time as _time, urllib.request
    from pymatgen.core import Lattice, Structure
    cache = CACHE_HOME / "_oqmd" / "pages"; cache.mkdir(parents=True, exist_ok=True)

    def fetch(offset):
        f = cache / f"offset_{offset:08d}.json"
        if f.exists():
            return _json.loads(f.read_text())
        url = f"{OQMD_API}?limit={page}&offset={offset}&fields={OQMD_FIELDS}&filter={OQMD_FILTER}"
        # oqmd.org answers 502 for stretches of minutes; wait it out rather than abort
        for attempt in range(10):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    d = _json.loads(r.read().decode())
                f.write_text(_json.dumps(d)); return d
            except Exception as e:
                err = e; _time.sleep(min(300, 15 * 2 ** attempt))
        print(f"  oqmd: page offset={offset} skipped after 10 attempts ({err})", file=sys.stderr)
        return {"data": [], "meta": {}}

    first = fetch(0)
    if not first.get("meta"):
        raise RuntimeError("OQMD API unreachable for the first page; try again later")
    total = int(first["meta"]["data_available"])
    n_pages = max(1, n_target // page); stride = max(page, total // n_pages)
    offsets = [i * stride for i in range(n_pages)]
    print(f"  oqmd: {total:,} entries with natoms<=40; sampling {n_pages} pages of {page} at stride {stride:,}")
    rows, bad = [], 0
    for i, off in enumerate(offsets, 1):
        d = fetch(off)
        for e in d.get("data", []):
            try:
                species, coords = [], []
                for s in e["sites"]:                      # "Ce @ 0.342 0.342 0.25"
                    el, xyz = s.split("@"); species.append(el.strip().split()[0])
                    coords.append([float(v) for v in xyz.split()])
                structure = Structure(Lattice(e["unit_cell"]), species, coords)
            except Exception:
                bad += 1; continue
            stab = _num(e.get("stability"))
            rows.append(dict(
                id=f"oqmd-{e['entry_id']}", mpid=None, structure=structure,
                formation_energy_per_atom=e.get("delta_e"), band_gap=e.get("band_gap"),
                band_gap_mbj=None, bulk_modulus=None,
                # OQMD stability is the hull distance; entries that define the hull can
                # come out slightly negative, which MP's E-hull never does -> floor at 0
                energy_above_hull=(max(stab, 0.0) if stab is not None else None),
                total_magnetization=None, stability_raw=stab))
        if i % 25 == 0:
            print(f"    oqmd pages {i}/{n_pages}  rows {len(rows):,}", flush=True)
    if bad:
        print(f"  oqmd: {bad} entries with unparseable sites skipped", file=sys.stderr)
    return rows, dict(source="oqmd:formationenergy", n=len(rows), stride=stride, total_available=total)


# ----------------------------------------------------------------------------- MP corpus
def load_mp_index():
    """{mp_id: record} for the 154,875 MP structures of the released corpus, plus a
    (reduced formula, nsites) -> [mp_id] prefilter for structure matching."""
    from pymatgen.core import Composition
    excluded = qc.excluded_materials()
    recs, pre = {}, defaultdict(list)
    with open(MP_STRUCTURES / "index.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r["mp_id"] in excluded:
                continue
            recs[r["mp_id"]] = r
            try:
                key = (Composition(r["formula"]).reduced_formula, int(r["nsites"]))
            except Exception:
                continue
            pre[key].append(r["mp_id"])
    return recs, pre


def match_against_mp(structure, pre, cache, matcher):
    """The MP id whose structure StructureMatcher deems equivalent, or None.

    Candidates are limited to MP entries with the same reduced formula and site count;
    the CIFs are parsed lazily and cached, so a run touches only the few hundred MP
    structures that share a formula with an unmatched external one."""
    from pymatgen.core import Structure
    key = (structure.composition.reduced_formula, len(structure))
    for mp_id in pre.get(key, ()):
        if mp_id not in cache:
            try:
                cache[mp_id] = Structure.from_file(MP_STRUCTURES / f"{mp_id}.cif")
            except Exception:
                cache[mp_id] = None
        s = cache[mp_id]
        if s is not None and matcher.fit(structure, s):
            return mp_id
    return None


def relabel_bulk_modulus(out, jsonl):
    """Replace bulk_modulus (stored as JARVIS's Voigt bound K_V by runs before 2026-11-03) with K_VRH from the
    elastic tensor, in place; records, matching and every other label are untouched. Logs the change."""
    import os
    from jarvis.db.figshare import data
    cwd = os.getcwd(); (CACHE_HOME / "_jarvis").mkdir(parents=True, exist_ok=True); os.chdir(CACHE_HOME / "_jarvis")
    try:
        d = data("dft_3d")
    finally:
        os.chdir(cwd)
    vrh = {str(e["jid"]): k_vrh(e.get("elastic_tensor")) for e in d}
    kv = {str(e["jid"]): _num(e.get("bulk_modulus_kv")) for e in d}
    recs = [json.loads(l) for l in open(jsonl)]
    old, new, diffs = 0, 0, []
    for r in recs:
        if r["bulk_modulus"] is not None:
            old += 1
        v = _bounded("bulk_modulus", vrh.get(r["mp_id"]))
        if v is not None and kv.get(r["mp_id"]) is not None:
            diffs.append(kv[r["mp_id"]] - v)
        r["bulk_modulus"] = v
        new += v is not None
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(jsonl)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("target_coverage", {})["bulk_modulus"] = new
    meta.setdefault("target_derivation", {})["bulk_modulus"] = NINE["bulk_modulus"]
    meta["bulk_modulus_relabelled"] = dict(date=str(date.today()), n_before_voigt=old, n_after_vrh=new,
                                           mean_kv_minus_kvrh_gpa=round(float(sum(diffs) / len(diffs)), 3) if diffs else None)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"bulk_modulus relabelled K_V -> K_VRH: {old:,} -> {new:,} labelled records; "
          f"K_V - K_VRH mean {meta['bulk_modulus_relabelled']['mean_kv_minus_kvrh_gpa']} GPa over {len(diffs):,} records")
    return 0


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["matminer", "jarvis-tools", "oqmd"], default="matminer")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: $MLIP_PROBES_ROOT/structures_external/jarvis_dft_3d, or .../oqmd for --source oqmd")
    ap.add_argument("--oqmd-n", type=int, default=40000, help="OQMD sample size (stride-sampled)")
    ap.add_argument("--no-match", action="store_true",
                    help="skip StructureMatcher for rows without a cited MP id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--relabel-bulk-modulus", action="store_true",
                    help="rewrite bulk_modulus in an existing jarvis-tools index.jsonl as K_VRH from the elastic tensor "
                         "(earlier runs stored the Voigt bound bulk_modulus_kv); touches no other field, then exits")
    args = ap.parse_args()

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    out = args.out_dir or (CACHE_HOME / ("oqmd" if args.source == "oqmd" else "jarvis_dft_3d"))
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "index.jsonl"
    done = F._load_done_ids(jsonl) if jsonl.exists() else set()

    t0 = time.time()
    if args.relabel_bulk_modulus:
        return relabel_bulk_modulus(out, jsonl)
    if args.source == "matminer":
        rows, meta = load_matminer()
    elif args.source == "jarvis-tools":
        rows, meta = load_jarvis_tools()
    else:
        rows, meta = load_oqmd(args.oqmd_n)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{meta['source']}: {len(rows):,} structures loaded in {time.time()-t0:.0f}s"
          f"{f', {len(done):,} already written' if done else ''}")

    mp_recs, pre = load_mp_index()
    matcher = StructureMatcher(primitive_cell=True, scale=True, attempt_supercell=False)
    cif_cache: dict = {}

    counts = defaultdict(int); n_written = 0; t0 = time.time()
    with open(jsonl, "a") as fh:
        for i, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            s = r["structure"]
            # relation to the MP corpus
            in_corpus = bool(r["mpid"] and r["mpid"] in mp_recs)
            match = None
            if not in_corpus and not args.no_match:
                match = match_against_mp(s, pre, cif_cache, matcher)
            counts["in_mp_corpus" if in_corpus else ("structure_match_mp" if match else "disjoint")] += 1

            try:
                sga = SpacegroupAnalyzer(s, symprec=0.1)
                crys, spg = sga.get_crystal_system().capitalize(), sga.get_space_group_symbol()
            except Exception:
                crys, spg = None, None
            vol = float(s.volume)
            tm = _num(r["total_magnetization"])
            bg = _bounded("band_gap", r["band_gap"])
            if bg is not None and 0 < bg <= 0.05:
                counts["gap_in_(0,0.05]"] += 1

            rec = {
                "mp_id":                     r["id"],                 # the key every stage reads
                "formula":                   s.composition.reduced_formula,
                "cif_file":                  str(out / f"{r['id']}.cif"),
                "nsites":                    len(s),
                "crystal_system":            crys,
                "space_group":               spg,
                "volume":                    round(vol, 3),
                "energy_above_hull":         _bounded("energy_above_hull", r["energy_above_hull"]),
                "formation_energy_per_atom": _bounded("formation_energy_per_atom", r["formation_energy_per_atom"]),
                "band_gap":                  bg,
                "bulk_modulus":              _bounded("bulk_modulus", r["bulk_modulus"]),
                "total_magnetization":       tm,
                "density":                   _bounded("density", float(s.density)),
                # provenance -- extra keys are ignored by 02 and mlipprobes.common
                "source":                    meta["source"],
                "band_gap_mbj":              _num(r["band_gap_mbj"]),
                "stability_raw":             r.get("stability_raw"),
                "mpid_reported":             r["mpid"],
                "in_mp_corpus":              in_corpus,
                "structure_match_mp":        match,
            }
            s.to(filename=rec["cif_file"])
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            n_written += 1
            if i % 1000 == 0:
                print(f"  {i:,}/{len(rows):,}  {time.time()-t0:.0f}s  "
                      f"corpus={counts['in_mp_corpus']} matched={counts['structure_match_mp']} "
                      f"disjoint={counts['disjoint']}", flush=True)

    # coverage of the nine paper targets, on what was written
    recs = [json.loads(l) for l in open(jsonl)]
    cov = {
        "formation_energy_per_atom": sum(r["formation_energy_per_atom"] is not None for r in recs),
        "energy_above_hull":         sum(r["energy_above_hull"] is not None for r in recs),
        "band_gap":                  sum(r["band_gap"] is not None for r in recs),
        "density":                   sum(r["density"] is not None for r in recs),
        "bulk_modulus":              sum(r["bulk_modulus"] is not None for r in recs),
        "mag_density":               sum(r["total_magnetization"] is not None for r in recs),
        "crystal_system":            sum(r["crystal_system"] is not None for r in recs),
        "metal_insulator":           sum(r["band_gap"] is not None for r in recs),
        "magnetic_label":            sum(r["total_magnetization"] is not None for r in recs),
    }
    meta.update(dict(
        date=str(date.today()), n_written=len(recs),
        functional=("PBE" if args.source == "oqmd" else "OptB88vdW"),
        relation_to_mp={k: sum(1 for r in recs if (r["in_mp_corpus"] if k == "in_mp_corpus"
                        else (r["structure_match_mp"] is not None) if k == "structure_match_mp"
                        else (not r["in_mp_corpus"] and r["structure_match_mp"] is None)))
                        for k in ("in_mp_corpus", "structure_match_mp", "disjoint")},
        target_coverage=cov, target_derivation=NINE,
        metal_gap_caveat=dict(gaps_in_0_to_0p05=sum(1 for r in recs if r["band_gap"] is not None and 0 < r["band_gap"] <= 0.05),
                              exact_zero=sum(1 for r in recs if r["band_gap"] == 0),
                              note="OptB88vdW metals carry small positive gaps; MP's band_gap==0 rule needs a threshold here"),
        matcher="StructureMatcher(primitive_cell=True, scale=True) on same reduced formula + nsites",
    ))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {n_written:,} new records ({len(recs):,} total) -> {out}")
    print("relation to MP corpus:", meta["relation_to_mp"])
    print("target coverage (of nine):")
    for k, v in cov.items():
        print(f"  {k:26s} {v:6,}  {'' if v else '-- NOT AVAILABLE from this source'}")
    print("metal-gap caveat:", meta["metal_gap_caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
