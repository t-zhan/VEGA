"""Compute the metrics from ``run_all_metrics.py`` with process-level parallelism."""

import argparse
import json
import multiprocessing as mp
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dcor
import h5py
import numpy as np
import perceptual_manifold_geometry as pmg
import torch
from IsoScore.IsoScore import IsoScore as _IsoScore
from scipy.spatial.distance import pdist
from skdim.id import TwoNN
from threadpoolctl import threadpool_limits
from torchdr.eval.neighborhood_preservation import neighborhood_preservation
from tqdm import tqdm

from visualize.autovla.decode_trajectory import load_codebook

ACTION_START = 151665
N_ACTION = 2048
BIG_N = 30000
MP_CONTEXT = mp.get_context("fork")
SPAWN_CONTEXT = mp.get_context("spawn")


# The official TLE wrapper prints once per sample. The proxy suppresses only
# those messages and remains process-local after fork().
_MUTE = threading.local()


class _StdoutProxy:
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        return len(text) if getattr(_MUTE, "on", False) else self.stream.write(text)

    def flush(self):
        if not getattr(_MUTE, "on", False):
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


sys.stdout = _StdoutProxy(sys.stdout)


def _iso(x):
    return float(_IsoScore(x))


def _twonn(x):
    return float(TwoNN().fit(x).dimension_)


def _tle(x):
    _MUTE.on = True
    value = float(pmg.estimate_intrinsic_dimension(x, method="TLE"))
    _MUTE.on = False
    return value


def _eff_rank(x):
    _, singular_values, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    p = singular_values / singular_values.sum()
    dimension = len(singular_values)
    return float((np.exp(-np.sum(p * np.log(p + 1e-12))) - 1) / (dimension - 1))


def _cos_dist(x, threads):
    z = x.astype(np.float32, copy=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    summed = z.sum(axis=0, dtype=np.float64)
    n = len(x)
    mean_cos = (float(summed @ summed) - n) / (n * (n - 1))
    return float(1.0 - mean_cos)


def _volume(x):
    z = x - x.mean(axis=0, keepdims=True)
    t, dimension = z.shape
    sign, logdet = np.linalg.slogdet(
        np.eye(t, dtype=z.dtype) + (dimension / t) * (z @ z.T)
    )
    return float(0.5 * logdet / np.log(2)) if sign > 0 else 0.0


def _d_stim(x):
    value = _tle(x)
    return value if np.isfinite(value) else None


def _dcor_blocked(x, y, block=4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(x)
    xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32, device=device)
    arow = torch.zeros(n, dtype=torch.float64, device=device)
    brow = torch.zeros(n, dtype=torch.float64, device=device)
    s_ab = s_aa = s_bb = 0.0

    for start in tqdm(range(0, n, block), desc="dCor blocked", leave=False):
        end = min(start + block, n)
        a = torch.cdist(xt[start:end], xt)
        b = torch.cdist(yt[start:end], yt)
        s_ab += float((a * b).sum(dtype=torch.float64))
        s_aa += float((a * a).sum(dtype=torch.float64))
        s_bb += float((b * b).sum(dtype=torch.float64))
        arow[start:end] = a.sum(dim=1, dtype=torch.float64)
        brow[start:end] = b.sum(dim=1, dtype=torch.float64)

    a_total, b_total = float(arow.sum()), float(brow.sum())
    ab_dot = float(arow @ brow)
    aa_dot = float(arow @ arow)
    bb_dot = float(brow @ brow)
    nf = float(n)
    dcov2 = s_ab / nf**2 - 2 * ab_dot / nf**3 + a_total * b_total / nf**4
    dvarx2 = s_aa / nf**2 - 2 * aa_dot / nf**3 + a_total**2 / nf**4
    dvary2 = s_bb / nf**2 - 2 * bb_dot / nf**3 + b_total**2 / nf**4
    denominator = np.sqrt(dvarx2 * dvary2)
    return float(np.sqrt(max(dcov2, 0.0) / denominator)) if denominator > 0 else 0.0


def _dcor_phys_emb(physical, embedding):
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
    if len(physical) > BIG_N:
        return _dcor_blocked(physical, embedding)
    return float(dcor.distance_correlation(physical, embedding))


def _knn_topk_blocked(x, k_max, block=4096):
    n = len(x)
    xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device="cuda")
    indices = torch.empty((n, k_max), dtype=torch.int64)
    for start in tqdm(range(0, n, block), desc="kNN blocked", leave=False):
        end = min(start + block, n)
        distances = torch.cdist(xt[start:end], xt)
        distances[
            torch.arange(end - start, device="cuda"),
            torch.arange(start, end, device="cuda"),
        ] = float("inf")
        indices[start:end] = distances.topk(k_max, dim=1, largest=False).indices.cpu()
    return indices


def _knn_phys_emb(physical, embedding, ks=(1, 5, 10, 50)):
    if len(physical) > BIG_N:
        k_max = max(ks)
        physical_knn = _knn_topk_blocked(physical, k_max)
        embedding_knn = _knn_topk_blocked(embedding, k_max)
        return {
            f"kNN_k{k}": float(
                (
                    physical_knn[:, :k].unsqueeze(2)
                    == embedding_knn[:, :k].unsqueeze(1)
                )
                .any(dim=2)
                .sum(dim=1)
                .double()
                .mean()
                / k
            )
            for k in ks
        }

    results = {}
    for k in ks:
        values = neighborhood_preservation(
            physical, embedding, K=k, return_per_sample=True
        )
        if hasattr(values, "numpy"):
            values = values.numpy()
        results[f"kNN_k{k}"] = float(np.mean(values))
    return results


# Most process workers read large arrays inherited through Linux fork/COW.
# Per-sample IsoScore and kNN use spawn and receive sample arrays explicitly.
_STATE = {}
_THREAD_LIMITER = None


def _init_worker(threads):
    global _THREAD_LIMITER
    _THREAD_LIMITER = threadpool_limits(limits=threads)
    torch.set_num_threads(threads)


def _init_spawn_worker(threads):
    _init_worker(threads)
    torch.set_num_interop_threads(1)


def _pool_map(worker, count, processes, threads, desc):
    chunksize = max(1, count // (processes * 8))
    with ProcessPoolExecutor(
        max_workers=processes,
        mp_context=MP_CONTEXT,
        initializer=_init_worker,
        initargs=(threads,),
    ) as executor:
        return list(
            tqdm(
                executor.map(worker, range(count), chunksize=chunksize),
                total=count,
                desc=desc,
                unit="spl",
            )
        )


def _iso_sample_worker(sample):
    return _iso(sample)


def _d_stim_sample_worker(sample):
    return _d_stim(sample)


def _spawn_map(worker, samples, count, processes, threads, desc, chunksize=1):
    with ProcessPoolExecutor(
        max_workers=processes,
        mp_context=SPAWN_CONTEXT,
        initializer=_init_spawn_worker,
        initargs=(threads,),
    ) as executor:
        return list(
            tqdm(
                executor.map(worker, samples, chunksize=chunksize),
                total=count,
                desc=desc,
                unit="spl",
            )
        )


def _summary(values):
    values = np.asarray([value for value in values if value is not None])
    return [float(values.mean()), float(values.std())]


def _hidden_worker(index):
    return _STATE["metric"](_STATE["hidden"][index])


def _first_worker(index):
    token_ids = _STATE["token_ids"][index] - _STATE["offset"]
    if not ((0 <= token_ids) & (token_ids < len(_STATE["embedding"]))).all():
        return None
    if _STATE["skip_errors"]:
        try:
            return _STATE["metric"](_STATE["embedding"][token_ids])
        except Exception:
            return None
    return _STATE["metric"](_STATE["embedding"][token_ids])


def _dcor_worker(index):
    token_ids = _STATE["token_ids"][index] - ACTION_START
    if not ((0 <= token_ids) & (token_ids < N_ACTION)).all():
        return None
    if len(np.unique(token_ids)) < 2:
        return None
    physical = _STATE["physical"][token_ids].astype(np.float32)
    embedding = (
        _STATE["embedding"][token_ids]
        if _STATE["use_index"]
        else _STATE["embedding"][index]
    )
    return _dcor_phys_emb(physical, embedding)


def _knn_sample_worker(sample):
    if sample is None:
        return None
    physical, embedding, k = sample
    values = neighborhood_preservation(
        physical, embedding, K=k, return_per_sample=True
    )
    if hasattr(values, "numpy"):
        values = values.numpy()
    return float(np.mean(values))


def _knn_samples(token_ids, embedding, physical, k, use_index):
    for index, ids in enumerate(token_ids):
        action_ids = ids - ACTION_START
        if not ((0 <= action_ids) & (action_ids < N_ACTION)).all():
            yield None
            continue
        if len(np.unique(action_ids)) < 2:
            yield None
            continue
        yield (
            np.ascontiguousarray(physical[action_ids], dtype=np.float32),
            np.ascontiguousarray(
                embedding[action_ids] if use_index else embedding[index]
            ),
            k,
        )


def _ps_cos_dist(x):
    return _cos_dist(x, _STATE["threads"])


def _per_sample_hidden(hidden, metric, processes, threads, desc, iso_threads):
    if metric is _iso:
        samples = (np.ascontiguousarray(sample) for sample in hidden)
        return _summary(
            _spawn_map(
                _iso_sample_worker,
                samples,
                len(hidden),
                processes,
                iso_threads,
                desc,
            )
        )
    if metric is _d_stim:
        samples = (np.ascontiguousarray(sample) for sample in hidden)
        return _summary(
            _spawn_map(
                _d_stim_sample_worker,
                samples,
                len(hidden),
                processes,
                threads,
                desc,
            )
        )
    _STATE.clear()
    _STATE.update(hidden=hidden, metric=metric, threads=threads)
    return _summary(_pool_map(_hidden_worker, len(hidden), processes, threads, desc))


def _per_sample_first(
    token_ids, embedding, metric, offset, processes, threads, desc, iso_threads,
    skip_errors=False,
):
    if metric is _iso:
        samples = (
            np.ascontiguousarray(embedding[ids - offset]) for ids in token_ids
        )
        return _summary(
            _spawn_map(
                _iso_sample_worker,
                samples,
                len(token_ids),
                processes,
                iso_threads,
                desc,
            )
        )
    if metric is _d_stim:
        samples = (
            np.ascontiguousarray(embedding[ids - offset]) for ids in token_ids
        )
        return _summary(
            _spawn_map(
                _d_stim_sample_worker,
                samples,
                len(token_ids),
                processes,
                threads,
                desc,
            )
        )
    _STATE.clear()
    _STATE.update(
        token_ids=token_ids,
        embedding=embedding,
        metric=metric,
        offset=offset,
        threads=threads,
        skip_errors=skip_errors,
    )
    return _summary(_pool_map(_first_worker, len(token_ids), processes, threads, desc))


def _per_sample_dcor(
    token_ids, embedding, physical, use_index, processes, threads, desc
):
    _STATE.clear()
    _STATE.update(
        token_ids=token_ids,
        embedding=embedding,
        physical=physical,
        use_index=use_index,
        threads=threads,
    )
    return _summary(_pool_map(_dcor_worker, len(token_ids), processes, threads, desc))


def _per_sample_knn(
    token_ids, embedding, physical, k, use_index, processes, threads, desc
):
    samples = _knn_samples(token_ids, embedding, physical, k, use_index)
    return _summary(
        _spawn_map(
            _knn_sample_worker,
            samples,
            len(token_ids),
            processes,
            threads,
            desc,
            chunksize=max(1, len(token_ids) // (processes * 8)),
        )
    )


def _inter_worker(task_index):
    kind, first, second = _STATE["tasks"][task_index]
    normalized = _STATE["normalized"]
    masks = _STATE["masks"]

    if kind == "intra":
        mask = masks[first]
        if mask.sum() < 2:
            return kind, None
        return kind, float(pdist(normalized[mask], metric="cosine").mean())

    mask1, mask2 = masks[first], masks[second]
    if mask1.sum() < 2 or mask2.sum() < 2:
        return kind, None
    pairs = min(5000, mask1.sum() * mask2.sum())
    n1 = min(mask1.sum(), int(np.sqrt(pairs)))
    n2 = min(mask2.sum(), int(np.sqrt(pairs)))
    rng = np.random.RandomState(42)
    rows1 = rng.choice(np.flatnonzero(mask1), size=n1, replace=False)
    rows2 = rng.choice(np.flatnonzero(mask2), size=n2, replace=False)
    return kind, float((1 - normalized[rows1] @ normalized[rows2].T).mean())


def _inter_intra(hidden, physical, processes, threads):
    forward, lateral = physical[:, 0], physical[:, 1]
    forward33, forward67 = np.percentile(forward, (33.3, 66.7))
    lateral33, lateral67 = np.percentile(lateral, (33.3, 66.7))
    speed = np.full(len(physical), "mid")
    speed[forward >= forward67] = "high"
    speed[forward <= forward33] = "low"
    direction = np.full(len(physical), "straight")
    direction[lateral >= lateral67] = "left"
    direction[lateral <= lateral33] = "right"
    strata = [
        f"{s}x{d}"
        for s in ("high", "mid", "low")
        for d in ("left", "straight", "right")
    ]
    masks = []
    for stratum in strata:
        s, d = stratum.split("x")
        masks.append((speed == s) & (direction == d))
    tasks = [("intra", i, -1) for i in range(len(strata))]
    tasks += [
        ("inter", i, j)
        for i in range(len(strata))
        for j in range(i + 1, len(strata))
    ]

    _STATE.clear()
    _STATE.update(
        normalized=hidden / (np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-12),
        masks=masks,
        tasks=tasks,
        threads=threads,
    )
    values = _pool_map(_inter_worker, len(tasks), processes, threads, "InterIntra")
    intra = [value for kind, value in values if kind == "intra" and value is not None]
    inter = [value for kind, value in values if kind == "inter" and value is not None]
    return float(np.mean(inter) / np.mean(intra))


def _load_results(path):
    if not Path(path).exists():
        return {}
    with open(path) as stream:
        return json.load(stream)


def _save_results(results, path):
    with open(path, "w") as stream:
        json.dump(results, stream, indent=2)


def _h5_tag_parts(h5):
    tokens = Path(h5).stem.split("-")

    # drop suffix noise (anything containing "embeddings")
    tokens = [t for t in tokens if "embeddings" not in t]

    structural = [t for t in tokens if t in ("train", "val") or t.startswith(("epoch_", "loss_"))]
    mode = [t for t in tokens if not (t in ("train", "val") or t.startswith(("epoch_", "loss_")))]

    return structural + (["-".join(mode)] if mode else [])


def _tagged_output_path(output, h5):
    tag = "-".join(_h5_tag_parts(h5))
    path = Path(output)
    return path.with_name(f"{path.stem}-{tag}{path.suffix}")


def _compute(results, output, section, name, function):
    if name not in results.get(section, {}):
        results.setdefault(section, {})[name] = function()
        _save_results(results, output)
    return results[section][name]


def _print_section(results, section):
    print(section)
    for name, value in results[section].items():
        if isinstance(value, list) and len(value) == 2:
            print(f"  {name}: {value[0]:.4f} +/- {value[1]:.4f}")
        elif isinstance(value, float):
            print(f"  {name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument(
        "--codebook", default="third_party/AutoVLA/codebook_cache/agent_vocab.pkl"
    )
    parser.add_argument(
        "--output", default="outputs/autovla/analysis/geo_metrics_mp.json"
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument(
        "--threads",
        type=int,
        default=12,
        help="Threads used by non-IsoScore metric functions inside each process",
    )
    parser.add_argument(
        "--iso-threads",
        type=int,
        default=2,
        help="Threads used by IsoScore (3584x3584 SVD) – keep low to avoid contention",
    )
    args = parser.parse_args()
    output = _tagged_output_path(args.output, args.h5)

    global _THREAD_LIMITER
    _THREAD_LIMITER = threadpool_limits(limits=args.threads)
    torch.set_num_threads(args.threads)

    print(f"Loading {args.h5}")
    print(f"Output {output}")
    with h5py.File(args.h5, "r") as h5:
        first_embedding = h5["first_embed"][:]
        last_hidden = h5["last_hidden"][:]
        token_ids = h5["token_ids"][:]
        text_first_embedding = h5["text_first_embed"][:]
        text_hidden = h5["text_hidden"][:]
        text_token_ids = h5["text_token_ids"][:]

    last_flat = last_hidden.reshape(-1, last_hidden.shape[-1])
    text_flat = text_hidden.reshape(-1, text_hidden.shape[-1])
    text_first_observed = text_first_embedding[np.unique(text_token_ids)]
    physical = load_codebook(Path(args.codebook))[:, -1].mean(dim=1).numpy().astype(np.float32)
    observed_action_ids = np.unique(token_ids - ACTION_START)
    action_first_observed = first_embedding[observed_action_ids]
    physical_observed = physical[observed_action_ids]
    physical_flat = physical[(token_ids - ACTION_START).ravel()]
    results = _load_results(output)
    results["h5_file"] = str(Path(args.h5).resolve())
    _save_results(results, output)

    global_metrics = [
        ("IsoScore", _iso),
        ("EffRank", _eff_rank),
        ("TwoNN", _twonn),
        ("D_world", _tle),
        ("Cosine_dist", lambda x: _cos_dist(x, args.threads)),
    ]
    for section, data in (
        ("action_first", action_first_observed),
        ("action_last", last_flat),
        ("text_first", text_first_observed),
        ("text_last", text_flat),
    ):
        for name, metric in global_metrics:
            _compute(results, output, section, name, lambda m=metric, x=data: m(x))
        _print_section(results, section)

    sample_metrics = [
        ("EffRank", _eff_rank),
        ("TwoNN", _twonn),
        ("Cosine_dist", _ps_cos_dist),
        ("V", _volume),
        ("IsoScore", _iso),
        ("D_stim", _d_stim),
    ]
    for section, ids, embedding, offset, prefix in (
        ("action_first_ps", token_ids, first_embedding, ACTION_START, "A1"),
        ("text_first_ps", text_token_ids, text_first_embedding, 0, "T1"),
    ):
        for name, metric in sample_metrics:
            skip_errors = name == "TwoNN"
            _compute(
                results,
                output,
                section,
                name,
                lambda m=metric, d=f"{prefix} ps {name}", s=skip_errors: _per_sample_first(
                    ids,
                    embedding,
                    m,
                    offset,
                    args.processes,
                    args.threads,
                    d,
                    args.iso_threads,
                    s,
                ),
            )
        _print_section(results, section)

    for section, hidden, prefix in (
        ("action_last_ps", last_hidden, "AL"),
        ("text_last_ps", text_hidden, "TL"),
    ):
        for name, metric in sample_metrics:
            _compute(
                results,
                output,
                section,
                name,
                lambda m=metric, d=f"{prefix} ps {name}": _per_sample_hidden(
                    hidden, m, args.processes, args.threads, d, args.iso_threads
                ),
            )
        _print_section(results, section)

    for section, embedding, use_index, prefix in (
        ("action_first_ps", first_embedding, True, "A1"),
        ("action_last_ps", last_hidden, False, "AL"),
    ):
        _compute(
            results,
            output,
            section,
            "dCor",
            lambda e=embedding, u=use_index, d=f"{prefix} ps dCor": _per_sample_dcor(
                token_ids,
                e,
                physical,
                u,
                args.processes,
                args.threads,
                d,
            ),
        )
        for k in (1, 5):
            _compute(
                results,
                output,
                section,
                f"kNN_k{k}",
                lambda e=embedding, u=use_index, kk=k, d=f"{prefix} ps kNN{k}": _per_sample_knn(
                    token_ids,
                    e,
                    physical,
                    kk,
                    u,
                    args.processes,
                    args.threads,
                    d,
                ),
            )
        _print_section(results, section)

    for section, embedding, points in (
        ("action_first", action_first_observed, physical_observed),
        ("action_last", last_flat, physical_flat),
    ):
        _compute(
            results,
            output,
            section,
            "dCor",
            lambda e=embedding, p=points: _dcor_phys_emb(p, e),
        )
        _compute(
            results,
            output,
            section,
            "InterIntra",
            lambda e=embedding, p=points: _inter_intra(
                e, p, args.processes, args.threads
            ),
        )
        if any(
            f"kNN_k{k}" not in results.get(section, {}) for k in (1, 5, 10, 50)
        ):
            results.setdefault(section, {}).update(_knn_phys_emb(points, embedding))
            _save_results(results, output)
        _print_section(results, section)


if __name__ == "__main__":
    main()
