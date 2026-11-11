"""mlipprobes/gnn.py — a small from-scratch message-passing GNN, trained per label budget.

The counterfactual for the label-efficiency comparison (stages/downstream/label_efficiency.py): "was
using a frozen foundation-model embedding even necessary, or could you just train your
own small model directly on your own labelled data?" No torch_geometric or similar graph
library is installed in this project, so the graph construction and message-passing
layers here are hand-written pure torch — deliberately small (the expense in ORB/UMA is
their equivariant machinery and 60M+ parameter scale, not graph size; materials here have
a ~20-atom median, so a 3-layer, 64-wide network trains in seconds to low-single-digit
minutes per run even on this project's CPU-only 8-core box).

Graph representation: element (atomic number) per node, a radius neighbour list (default
cutoff 6.0 Angstrom, capped at 16 nearest neighbours per atom -- verified against mp-1
(elemental Cs, BCC): its nearest-neighbour distance is 5.42 Angstrom, so a 5.0 Angstrom
cutoff would leave it with zero edges) via pymatgen's get_neighbor_list, packed once into
a single ragged archive so it is built exactly once per material regardless of how many
budgets/targets/repeats reuse it.

Pack format mirrors the project's established ragged-array convention (see
pooling_future_work/POOLING_PROJECT_PLAN.md's "Pack format" note): concatenated per-atom /
per-edge arrays plus an offsets array marking each material's slice.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CUTOFF = 6.0
MAX_NEIGHBORS = 16


# --------------------------------------------------------------------------- graph build
def build_graph(structure, cutoff: float = CUTOFF, max_neighbors: int = MAX_NEIGHBORS):
    """One material -> (z [N] uint8 atomic numbers, edge_src/edge_dst [E] int32 local
    node indices, edge_dist [E] float32 Angstrom). Caps each atom's edge list to its
    `max_neighbors` nearest (by distance) to bound graph size on densely-coordinated
    structures without changing the cutoff globally."""
    n = len(structure)
    z = np.array([site.specie.Z for site in structure], dtype=np.uint8)
    ci, pi, _offsets, dists = structure.get_neighbor_list(r=cutoff)
    if len(ci) == 0:
        # no neighbours within cutoff (shouldn't happen post mp-1 fix, but stay safe)
        return z, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    # cap to max_neighbors nearest per source atom
    order = np.lexsort((dists, ci))
    ci, pi, dists = ci[order], pi[order], dists[order]
    keep = np.ones(len(ci), dtype=bool)
    counts = np.zeros(n, dtype=np.int64)
    for i in range(len(ci)):
        c = ci[i]
        if counts[c] >= max_neighbors:
            keep[i] = False
        else:
            counts[c] += 1
    return z, ci[keep].astype(np.int32), pi[keep].astype(np.int32), dists[keep].astype(np.float32)


def pack_graphs(mp_ids: list[str], structures_dir: Path, cache_path: Path,
                cutoff: float = CUTOFF, max_neighbors: int = MAX_NEIGHBORS,
                n_jobs: int = 8, chunk: int = 2000) -> None:
    """Build + cache graphs for every id in mp_ids (skips ids already present if
    cache_path exists, so a partial/interrupted run resumes). Idempotent, matching
    every other stage in this pipeline."""
    from pymatgen.core import Structure
    from joblib import Parallel, delayed

    have = set()
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        have = set(d["mp_ids"].astype(str))
        print(f"  {cache_path.name}: {len(have):,} materials already packed")

    todo = [m for m in mp_ids if m not in have]
    if not todo:
        print("  nothing to pack"); return
    print(f"  packing {len(todo):,} materials (cutoff={cutoff}, max_neighbors={max_neighbors})")

    def _one(mid):
        try:
            s = Structure.from_file(str(structures_dir / f"{mid}.cif"))
            z, es, ed, dist = build_graph(s, cutoff, max_neighbors)
            return mid, z, es, ed, dist
        except Exception as ex:
            return mid, None, None, None, str(ex)

    t0 = time.time()
    results = []
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        out = Parallel(n_jobs=n_jobs)(delayed(_one)(m) for m in batch)
        results.extend(out)
        done = i + len(batch)
        rate = done / (time.time() - t0)
        print(f"    {done:,}/{len(todo):,}  ({rate:.0f}/s, ETA {(len(todo)-done)/rate/60:.1f} min)",
             end="\r", flush=True)
    print()

    ok = [(m, z, es, ed, dist) for m, z, es, ed, dist in results if z is not None]
    if len(ok) < len(results):
        failed = [(m, err) for m, z, _es, _ed, err in results if z is None]
        print(f"  {len(failed)} materials failed graph construction (skipped): "
             f"{failed[:3]}{' ...' if len(failed) > 3 else ''}")

    node_z = np.concatenate([z for _, z, *_r in ok]) if ok else np.zeros(0, np.uint8)
    node_offsets = np.concatenate([[0], np.cumsum([len(z) for _, z, *_r in ok])])
    edge_src = np.concatenate([es for _, _z, es, *_r in ok]) if ok else np.zeros(0, np.int32)
    edge_dst = np.concatenate([ed for _, _z, _es, ed, _d in ok]) if ok else np.zeros(0, np.int32)
    edge_dist = np.concatenate([d for _, _z, _es, _ed, d in ok]) if ok else np.zeros(0, np.float32)
    edge_offsets = np.concatenate([[0], np.cumsum([len(es) for _, _z, es, *_r in ok])])
    new_mp_ids = np.array([m for m, *_r in ok], dtype=object)

    if have:
        prev = np.load(cache_path, allow_pickle=True)
        node_z = np.concatenate([prev["node_z"], node_z])
        node_offsets = np.concatenate([prev["node_offsets"][:-1],
                                       prev["node_offsets"][-1] + node_offsets])
        edge_src = np.concatenate([prev["edge_src"], edge_src])
        edge_dst = np.concatenate([prev["edge_dst"], edge_dst])
        edge_dist = np.concatenate([prev["edge_dist"], edge_dist])
        edge_offsets = np.concatenate([prev["edge_offsets"][:-1],
                                       prev["edge_offsets"][-1] + edge_offsets])
        new_mp_ids = np.concatenate([prev["mp_ids"], new_mp_ids])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, node_z=node_z, node_offsets=node_offsets,
             edge_src=edge_src, edge_dst=edge_dst, edge_dist=edge_dist,
             edge_offsets=edge_offsets, mp_ids=new_mp_ids)
    print(f"  wrote {cache_path} ({len(new_mp_ids):,} materials total)")


class GraphCache:
    """Read-only view over a packed archive; slices out a material's graph by mp_id."""

    def __init__(self, cache_path: Path):
        d = np.load(cache_path, allow_pickle=True)
        self.node_z, self.node_offsets = d["node_z"], d["node_offsets"]
        self.edge_src, self.edge_dst = d["edge_src"], d["edge_dst"]
        self.edge_dist, self.edge_offsets = d["edge_dist"], d["edge_offsets"]
        self.pos = {m: i for i, m in enumerate(d["mp_ids"].astype(str))}

    def __contains__(self, mp_id):
        return mp_id in self.pos

    def get(self, mp_id):
        """Edge indices in the packed archive are already local to each material
        (0-based, as build_graph/pack_graphs produced them) -- no offset subtraction
        needed here, only the node slice itself uses the offsets."""
        i = self.pos[mp_id]
        ns, ne = self.node_offsets[i], self.node_offsets[i + 1]
        es, ee = self.edge_offsets[i], self.edge_offsets[i + 1]
        return (self.node_z[ns:ne],
               self.edge_src[es:ee].astype(np.int64),
               self.edge_dst[es:ee].astype(np.int64),
               self.edge_dist[es:ee])


# --------------------------------------------------------------------------- model
class _RBF(nn.Module):
    """Gaussian radial basis expansion of a scalar edge distance -> a vector,
    the standard way to hand a continuous distance to an MLP (a raw scalar makes
    it hard for a small MLP to learn a non-monotonic response)."""

    def __init__(self, n=16, cutoff=CUTOFF):
        super().__init__()
        centers = torch.linspace(0, cutoff, n)
        self.register_buffer("centers", centers)
        self.width = cutoff / n

    def forward(self, d):
        return torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width ** 2))


class MPLayer(nn.Module):
    """One message-passing round: edge message = MLP(h_src, h_dst, rbf(dist)),
    sum-aggregated into each destination node, then a residual node update."""

    def __init__(self, hidden, n_rbf):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * hidden + n_rbf, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_src, edge_dst, rbf):
        m = self.msg(torch.cat([h[edge_src], h[edge_dst], rbf], dim=-1))
        agg = torch.zeros_like(h).index_add_(0, edge_dst, m)
        return self.norm(h + self.upd(agg))


class SmallGNN(nn.Module):
    def __init__(self, n_out: int, hidden: int = 64, n_layers: int = 3, n_rbf: int = 16,
                cutoff: float = CUTOFF):
        super().__init__()
        self.embed = nn.Embedding(119, hidden)   # index by atomic number, Z<=118
        self.rbf = _RBF(n_rbf, cutoff)
        self.layers = nn.ModuleList([MPLayer(hidden, n_rbf) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))

    def forward(self, z, edge_src, edge_dst, edge_dist, batch, n_graphs):
        h = self.embed(z.long())
        rbf = self.rbf(edge_dist)
        for layer in self.layers:
            h = layer(h, edge_src, edge_dst, rbf)
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(
            0, batch, torch.ones(h.shape[0], device=h.device)).clamp(min=1).unsqueeze(-1)
        pooled = pooled / counts
        return self.head(pooled)


# --------------------------------------------------------------------------- batching
def _collate(mp_ids, cache: GraphCache, device):
    zs, es_l, ed_l, dist_l, batch_l = [], [], [], [], []
    node_offset = 0
    for gi, m in enumerate(mp_ids):
        z, es, ed, dist = cache.get(m)
        zs.append(z); dist_l.append(dist)
        es_l.append(es + node_offset); ed_l.append(ed + node_offset)
        batch_l.append(np.full(len(z), gi, dtype=np.int64))
        node_offset += len(z)
    z_t = torch.as_tensor(np.concatenate(zs), device=device)
    es_t = torch.as_tensor(np.concatenate(es_l), device=device)
    ed_t = torch.as_tensor(np.concatenate(ed_l), device=device)
    dist_t = torch.as_tensor(np.concatenate(dist_l), dtype=torch.float32, device=device)
    batch_t = torch.as_tensor(np.concatenate(batch_l), device=device)
    return z_t, es_t, ed_t, dist_t, batch_t


# --------------------------------------------------------------------------- training
def train_gnn(mp_ids_train, y_train, mp_ids_val, y_val, task: str, cache: GraphCache,
              n_classes: int = 1, hidden: int = 64, n_layers: int = 3, lr: float = 1e-3,
              max_epochs: int = 200, patience: int = 15, batch_size: int = 256,
              seed: int = 42, device: str = "cpu", verbose: bool = False):
    """task: 'reg' (n_classes ignored, 1 output) or 'clf' (n_classes outputs, CE loss).
    Early-stops on the held-out val set (carved from the training budget by the
    caller — never the real test set). Returns (model, best_val_score)."""
    torch.manual_seed(seed)
    n_out = 1 if task == "reg" else n_classes
    model = SmallGNN(n_out, hidden, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    yt_train = torch.as_tensor(y_train, dtype=torch.float32 if task == "reg" else torch.long,
                               device=device)
    yt_val = torch.as_tensor(y_val, dtype=torch.float32 if task == "reg" else torch.long,
                             device=device)

    rng = np.random.default_rng(seed)
    n = len(mp_ids_train)
    best_val, best_state, bad_epochs = float("inf") if task == "reg" else -float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_ids = [mp_ids_train[j] for j in idx]
            z, es, ed, dist, batch = _collate(batch_ids, cache, device)
            out = model(z, es, ed, dist, batch, len(idx))
            yb = yt_train[idx]
            loss = F.mse_loss(out.squeeze(-1), yb) if task == "reg" else F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            z, es, ed, dist, batch = _collate(list(mp_ids_val), cache, device)
            out = model(z, es, ed, dist, batch, len(mp_ids_val))
            if task == "reg":
                pred = out.squeeze(-1)
                ss_res = ((pred - yt_val) ** 2).sum()
                ss_tot = ((yt_val - yt_val.mean()) ** 2).sum().clamp(min=1e-8)
                val_score = float(1 - ss_res / ss_tot)  # R2
            else:
                pred = out.argmax(-1)
                val_score = float((pred == yt_val).float().mean())  # accuracy (cheap proxy; F1 at eval time)

        if best_state is None or val_score > best_val:
            best_val, best_state, bad_epochs = val_score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch}: val_score={val_score:.3f} (best={best_val:.3f})")
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def predict_gnn(model, mp_ids, cache: GraphCache, task: str, device: str = "cpu",
                batch_size: int = 512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(mp_ids), batch_size):
            batch_ids = mp_ids[i:i + batch_size]
            z, es, ed, dist, batch = _collate(batch_ids, cache, device)
            out = model(z, es, ed, dist, batch, len(batch_ids))
            preds.append(out.squeeze(-1).numpy() if task == "reg" else out.argmax(-1).numpy())
    return np.concatenate(preds)
