#!/usr/bin/env python3
"""
stages/extract/extract_mace.py — pooled per-layer node read-outs of MACE-OMAT-0 (medium), the
equivariant model trained on OMat24 only, for the matched-training-data comparison with
ORB-v3-omat (stages/controls/matched_models.py). Runs in its own environment (envs/mace: mace-torch,
pymatgen), because MACE pins its own e3nn/torch stack.

MACE medium has two interaction/product blocks. Forward hooks on `model.products[i]` capture the
node features after each: layer 0 is 128x0e + 128x1o (128 scalars + 128 vectors = 512 numbers per
atom), layer 1 is 128x0e (scalars only). Three pooled read-outs per layer, mean over atoms:

  full      every component                              (512 / 128)
  scalar    the 128 l=0 channels                         (128 / 128)  rotation-invariant
  ang_norm  scalars + the norm of each l=1 vector        (256 / 128)  rotation-invariant

Writes <results-dir>/pooled/layer_{i}_{family}.npz with X [n, d] float32 and mp_ids, plus
manifest.json. Resumable: each worker writes a part file every 500 materials and materials already
in the part files are skipped on re-launch.

Usage
    uv run --project envs/mace python stages/extract/extract_mace.py \\
        --ids-file $MLIP_PROBES_ROOT/results/matched_data/subsample_ids.txt --results-dir $MLIP_PROBES_ROOT/results/mace_omat \\
        --num-workers 4 --torch-threads 2
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import json
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np

FAMILIES = ["full", "scalar", "ang_norm"]
MODEL = "medium-omat-0"
FLUSH_EVERY = 500      # materials per part file: what a killed worker can lose


def _pool(feats: dict) -> dict:
    """{(layer, family): pooled vector} from the hooked per-atom features."""
    out = {}
    for i, t in feats.items():
        a = t.float().cpu().numpy()                       # [N, d]
        n_scalar = 128
        out[(i, "full")] = a.mean(0)
        out[(i, "scalar")] = a[:, :n_scalar].mean(0)
        if a.shape[1] > n_scalar:
            # e3nn stores 128x1o channel by channel: [channel, 3 components]. Checked on rotated structures:
            # with this layout the norms change by ~1e-5 under a random rotation (0.7 with the other one).
            vec = a[:, n_scalar:].reshape(len(a), -1, 3)
            out[(i, "ang_norm")] = np.concatenate([a[:, :n_scalar].mean(0), np.linalg.norm(vec, axis=2).mean(0)])
        else:
            out[(i, "ang_norm")] = a[:, :n_scalar].mean(0)
    return out


def _worker(wid, ids, structures_dir, out_dir, torch_threads, q):
    import torch
    torch.set_num_threads(torch_threads)
    from mace.calculators import mace_mp
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    calc = mace_mp(model=MODEL, default_dtype="float32", device="cpu")
    model = calc.models[0]
    feats = {}
    for i, p in enumerate(model.products):
        p.register_forward_hook(lambda mod, inp, out, i=i: feats.__setitem__(i, out.detach()))
    adaptor = AseAtomsAdaptor()
    rows, n_done, n_part = {}, 0, 0
    t0 = time.time()

    def flush():
        """Write the pooled rows gathered so far as one part per (layer, family): a killed run keeps them.
        The layer_0_full part is written last, because the resume check reads that one."""
        nonlocal rows, n_part
        if not rows:
            return
        keys = sorted({kk for r in rows.values() for kk in r}, key=lambda kk: kk == (0, "full"))
        for (i, fam) in keys:
            ms = [m for m in rows if (i, fam) in rows[m]]
            tmp = out_dir / f"tmp_w{wid}_{n_part}_layer_{i}_{fam}.npz"
            np.savez(tmp, X=np.stack([rows[m][(i, fam)] for m in ms]).astype(np.float32), mp_ids=np.array(ms, dtype=object))
            tmp.rename(out_dir / f"part_w{wid}_{n_part}_layer_{i}_{fam}.npz")
        rows, n_part = {}, n_part + 1

    for k, m in enumerate(ids, 1):
        try:
            at = adaptor.get_atoms(Structure.from_file(str(structures_dir / f"{m}.cif"))); at.calc = calc
            feats.clear(); at.get_potential_energy()
            rows[m] = _pool(feats); n_done += 1
        except Exception as e:
            q.put((wid, "err", f"{m}: {e}"))
        if k % 200 == 0:
            q.put((wid, "progress", f"{k}/{len(ids)} ({(time.time()-t0)/k:.2f}s/mat)"))
        if k % FLUSH_EVERY == 0:
            flush()
    flush()
    q.put((wid, "done", n_done))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures-dir", default=str(P.STRUCTURES))
    ap.add_argument("--results-dir", default=str(P.RESULTS / "mace_omat"))
    ap.add_argument("--ids-file", required=True, help="one mp_id per line")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=2)
    a = ap.parse_args()
    os.environ.setdefault("TORCH_HOME", "~/.cache/torch")
    out = Path(a.results_dir) / "pooled"; out.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    done = set()
    for f in out.glob("part_*_layer_0_full.npz"):
        done |= set(np.load(f, allow_pickle=True)["mp_ids"].astype(str))
    todo = [m for m in ids if m not in done]
    died = False
    print(f"{len(ids):,} ids, {len(done):,} done, {len(todo):,} to extract with {a.num_workers} workers x {a.torch_threads} threads", flush=True)
    if todo:
        q = Queue(); procs = []
        chunks = [todo[i::a.num_workers] for i in range(a.num_workers)]
        stamp = int(time.time())
        for w, ch in enumerate(chunks):
            if ch:
                procs.append(Process(target=_worker, args=(f"{stamp}_{w}", ch, Path(a.structures_dir), out, a.torch_threads, q))); procs[-1].start()
        import queue as _queue
        live = len(procs)
        while live:
            try:
                wid, kind, msg = q.get(timeout=60)
            except _queue.Empty:
                if not any(p.is_alive() for p in procs):       # a worker died without reporting: do not wait for ever
                    print("  workers exited without finishing; re-launch to resume", flush=True); died = True; break
                continue
            if kind == "done":
                live -= 1; print(f"  worker {wid} done: {msg} materials", flush=True)
            elif kind == "progress":
                print(f"  worker {wid}: {msg}", flush=True)
            else:
                print(f"  worker {wid} ERROR {msg}", flush=True)
        for p in procs:
            p.join()
    # merge parts into one file per (layer, family), in the ids-file order
    merged = {}
    for f in sorted(out.glob("part_*.npz")):
        key = f.stem.split("_layer_")[1]
        z = np.load(f, allow_pickle=True)
        merged.setdefault(key, []).append((z["mp_ids"].astype(str), z["X"]))
    order = {m: i for i, m in enumerate(ids)}
    man = {"model": MODEL, "families": FAMILIES, "layers": []}
    for key, parts in merged.items():
        mids = np.concatenate([p[0] for p in parts]); X = np.concatenate([p[1] for p in parts])
        _, first = np.unique(mids, return_index=True)
        mids, X = mids[first], X[first]
        o = np.argsort([order.get(m, 1e9) for m in mids])
        np.savez(out / f"layer_{key}.npz", X=X[o], mp_ids=mids[o].astype(object))
        man["layers"].append({"file": f"layer_{key}.npz", "n": int(len(o)), "dim": int(X.shape[1])})
    (Path(a.results_dir) / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {len(merged)} pooled files to {out}")
    if died:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
