"""
Decode AutoVLA action tokens and visualize trajectories as a grid.

Reproduces AutoVLA's action_tokenizer.rollout() logic directly from the codebook,
without loading the full VLM.

Usage:
    # Select by speed + lateral direction (generates n subplots in one image)
    python src/analysis/autovla/decode_trajectory.py --split train --speed high --lateral left --n 6
    python src/analysis/autovla/decode_trajectory.py --split val  --speed mid  --lateral all  --n 4

    # Specify exact indices (comma-separated)
    python src/analysis/autovla/decode_trajectory.py --split train --idx 0,42,6974

    # Save output
    python src/analysis/autovla/decode_trajectory.py --speed high --n 6 --output out.png
"""

import argparse
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ── constants ──────────────────────────────────────────────────────────────────
ACTION_START_ID = 151665


# ── decoding ───────────────────────────────────────────────────────────────────

def load_codebook(pkl_path: Path) -> torch.Tensor:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return torch.tensor(data["token_all"]["veh"], dtype=torch.float32)  # (2048,6,4,2)


def _transform(pos_local, pos_now, head_now):
    c, s = torch.cos(head_now), torch.sin(head_now)
    return pos_local @ torch.tensor([[c, s], [-s, c]]) + pos_now


def rollout(action_tokens: torch.Tensor):
    """action_tokens: (T, 6, 4, 2) → pos (T+1,2), head (T+1,), substeps list."""
    T = action_tokens.shape[0]
    pos, head = torch.zeros(T + 1, 2), torch.zeros(T + 1)
    substeps = []
    for t in range(T):
        flat_g = _transform(action_tokens[t].reshape(-1, 2), pos[t], head[t])
        tok_g  = flat_g.reshape(6, 4, 2)
        substeps.append(tok_g)
        pos[t + 1]  = tok_g[-1].mean(0)
        diff         = tok_g[-1, 0] - tok_g[-1, 3]
        head[t + 1]  = torch.arctan2(diff[1], diff[0])
    return pos, head, substeps


def decode_frame(token_ids_row: np.ndarray, codebook: torch.Tensor):
    """Return pos (T+1,2), head (T+1,), substep_list, bin_ids (T,)."""
    bin_ids       = token_ids_row - ACTION_START_ID
    action_tokens = codebook[torch.tensor(bin_ids)]
    pos, head, subs = rollout(action_tokens)
    return pos.numpy(), head.numpy(), [s.numpy() for s in subs], bin_ids


# ── index selection ────────────────────────────────────────────────────────────

def _speed_stats(token_ids_all: np.ndarray, codebook: torch.Tensor):
    bins  = token_ids_all - ACTION_START_ID
    c6    = codebook[:, -1].mean(dim=1).numpy()          # (2048, 2)
    sfwd  = np.array([c6[bins[i], 0].mean() for i in range(len(bins))])
    slat  = np.array([c6[bins[i], 1].mean() for i in range(len(bins))])
    return sfwd, slat


def _batch_rollout_pos(token_ids_rows: np.ndarray, codebook: torch.Tensor) -> np.ndarray:
    """Batch rollout for N samples. Returns pos (N, T+1, 2) as numpy array.

    Runs the T-step loop once (T is small, typically 6), with all N samples
    processed in parallel via vectorised torch ops at each step.
    """
    bins = token_ids_rows - ACTION_START_ID          # (N, T)
    N, T = bins.shape
    tokens = codebook[torch.tensor(bins)]            # (N, T, 6, 4, 2)

    pos  = torch.zeros(N, T + 1, 2)
    head = torch.zeros(N, T + 1)

    for t in range(T):
        c = torch.cos(head[:, t])                   # (N,)
        s = torch.sin(head[:, t])
        R = torch.stack([torch.stack([c, s], -1),
                         torch.stack([-s, c], -1)], dim=-2)  # (N, 2, 2)
        flat = tokens[:, t].reshape(N, -1, 2)       # (N, 24, 2)
        flat_g = (flat @ R) + pos[:, t:t+1]         # (N, 24, 2)
        tok_g = flat_g.reshape(N, 6, 4, 2)
        pos[:, t + 1] = tok_g[:, -1].mean(dim=-2)   # (N, 2)
        diff = tok_g[:, -1, 0] - tok_g[:, -1, 3]    # (N, 2)
        head[:, t + 1] = torch.arctan2(diff[:, 1], diff[:, 0])

    return pos.numpy()                               # (N, T+1, 2)


def _straight_rmse(pos_all: np.ndarray) -> np.ndarray:
    """Compute least-squares line residual RMSE for each trajectory.

    Uses SVD (PCA) on centred trajectory points: the second right singular
    vector is perpendicular to the best-fit line; projecting onto it gives
    the signed perpendicular distances.

    Returns rmse (N,).
    """
    centered = pos_all - pos_all.mean(axis=1, keepdims=True)  # (N, T+1, 2)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)   # Vt: (N, 2, 2)
    perp = Vt[:, 1, :]                                        # (N, 2)
    residuals = (centered * perp[:, None, :]).sum(-1)          # (N, T+1)
    return np.sqrt((residuals ** 2).mean(axis=1))              # (N,)


def select_idx(token_ids_all, codebook, speed, lateral, n):
    """Return n row indices matching speed + lateral criteria."""
    sfwd, slat = _speed_stats(token_ids_all, codebook)
    S = len(sfwd)

    pool_size = S
    if speed == "high":
        pool = sfwd.argsort()[-pool_size:]
    elif speed == "low":
        pool = sfwd.argsort()[:pool_size]
    else:  # mid
        pool = np.argsort(np.abs(sfwd - np.median(sfwd)))[:pool_size]

    lat_pool = slat[pool]
    if lateral == "left":
        chosen = pool[lat_pool.argsort()[-n:][::-1]]
    elif lateral == "right":
        chosen = pool[lat_pool.argsort()[:n]]
    elif lateral == "straight":
        # Use SVD least-squares residual: smallest RMSE = most linear trajectory
        pos_pool = _batch_rollout_pos(token_ids_all[pool], codebook)
        rmse = _straight_rmse(pos_pool)
        chosen = pool[rmse.argsort()[:n]]
    else:  # all — evenly spaced by lateral displacement
        ordered = pool[lat_pool.argsort()]
        chosen  = ordered[np.linspace(0, len(ordered) - 1, min(n, len(ordered)), dtype=int)]

    return chosen.tolist(), sfwd, slat


# ── font-metric constants — all spacing derived from these ────────────────────
_FS_TICK    = 11.0   # pt  tick-label fontsize
_FS_LABEL   = 13.0   # pt  axis-label fontsize
_FS_CAPTION = 11.5   # pt  subplot caption fontsize
_FS_TITLE   = 16.0   # pt  figure title fontsize
_FS_CB      = 12.5   # pt  colorbar label fontsize
_FS_ANNOT   = 10.0   # pt  token-id annotation fontsize
_LS         = 1.35   # line-spacing multiplier

def _ph(fs): return fs * _LS / 72.0        # point → inches (line height)
def _pw(fs, n=1): return 0.55 * fs * n / 72.0  # char width × n chars → inches

# Vertical space consumed below subplot axes (x-ticks + x-label + caption + padding)
_BELOW_H = (_ph(_FS_TICK)    + 4/72          # x-tick labels
           + _ph(_FS_LABEL)  + 4/72          # x-axis label "x (m)"
           + _ph(_FS_CAPTION) * 2 + 0.14)   # two-line caption + buffer

# Horizontal space consumed to the left of subplot axes (y-label + y-ticks)
_LEFT_W  = (_ph(_FS_LABEL)                 # rotated ylabel (height → width)
           + 5/72
           + _pw(_FS_TICK, 5)              # ~5-char tick labels ("-2.5")
           + 0.14)

# Caption offset from the bottom of axes in points (places 2-line caption below xlabel)
_CAP_OFF_PT = (_FS_TICK * _LS + 5          # past x-tick labels
             + _FS_LABEL * _LS + 5         # past xlabel "x (m)"
             + 4)                          # small buffer

# Colorbar (vertical, right side): band width + gap + label width
_CB_BAND_W  = 0.20   # colorbar colour band width (in)
_CB_GAP     = 0.16   # gap between grid right edge and colorbar band

# TOP margin: two-line title + padding
_TOP_H = _ph(_FS_TITLE) * 2 + 0.24

# RIGHT margin: gap + colorbar band + tick labels + label (rotated) + padding
_RIGHT_W = (_CB_GAP + _CB_BAND_W
          + _pw(_FS_CB, 2) + 4/72          # 2-char tick numbers ("10")
          + _ph(_FS_CB) * 2 + 0.30)        # rotated label height × 2 + padding


# ── drawing ────────────────────────────────────────────────────────────────────

def _draw_subplot(ax, pos, head, substep_list, bin_ids, show_substeps, caption,
                  xlim=None, ylim=None, alternate=False):
    """Draw one decoded trajectory on ax.

    xlim / ylim: shared data limits for uniform axes across a grid.
    When provided, sub_w/sub_h must already equal (xlim range)/(ylim range)
    so set_aspect("equal") fills the axes without whitespace.

    Caption is placed at a fixed offset in *points* below the axes bottom so
    it never overlaps tick labels regardless of subplot aspect ratio.
    """
    T      = len(bin_ids)
    cmap   = plt.cm.plasma
    colors = [cmap(i / (T - 1)) for i in range(T)]

    dx_data = (xlim[1] - xlim[0]) if xlim else None
    arrow_len = max(0.4, dx_data * 0.012) if dx_data else 0.35

    ax.set_facecolor("#f5f5f7")
    ax.grid(True, color="white", linewidth=0.8, zorder=0)

    # Always: filled bbox of last substep per token (vehicle outline)
    for t, subs in enumerate(substep_list):
        poly = plt.Polygon(subs[-1][[0, 1, 2, 3]], closed=True,
                           facecolor=colors[t], edgecolor=colors[t],
                           alpha=0.22, linewidth=0.8, zorder=1)
        ax.add_patch(poly)

    # Optional: all 6-substep dashed outlines
    if show_substeps:
        for t, subs in enumerate(substep_list):
            for step_i in range(6):
                poly = plt.Polygon(subs[step_i][[0, 1, 2, 3]],
                                   fill=False, edgecolor=colors[t],
                                   linewidth=0.4, alpha=0.28, linestyle="--", zorder=1)
                ax.add_patch(poly)

    ax.plot(pos[:, 0], pos[:, 1], color="#333333", linewidth=1.4, zorder=3)
    ax.scatter(*pos[0], c="#27ae60", s=80, zorder=6, edgecolors="white", linewidths=1.1)

    for t in range(1, T + 1):
        ax.scatter(*pos[t], c=[colors[t-1]], s=40, zorder=5,
                   edgecolors="white", linewidths=0.7)
        va = "bottom" if not alternate or t % 2 == 1 else "top"
        dy = 3 if not alternate or t % 2 == 1 else -3
        ax.annotate(f"#{ACTION_START_ID + bin_ids[t-1]}", pos[t], fontsize=_FS_ANNOT, color="#333333",
                    ha="left", va=va, xytext=(3, dy),
                    textcoords="offset points", zorder=7)
        adx = arrow_len * np.cos(head[t])
        ady = arrow_len * np.sin(head[t])
        ax.annotate("", xy=(pos[t, 0]+adx, pos[t, 1]+ady), xytext=pos[t],
                    arrowprops=dict(arrowstyle="-|>", color=colors[t-1],
                                   lw=0.9, mutation_scale=8), zorder=6)

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
    else:
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
        xl, xr = ax.get_xlim(); yl, yh = ax.get_ylim()
        px = max(1.5, (xr-xl)*0.10); py = max(1.5, (yh-yl)*0.20)
        ax.set_xlim(xl-px, xr+px); ax.set_ylim(yl-py, yh+py)

    ax.set_xlabel("x (m)", fontsize=_FS_LABEL, labelpad=4)
    ax.set_ylabel("y (m)", fontsize=_FS_LABEL, labelpad=4)
    ax.tick_params(labelsize=_FS_TICK)
    # Caption: fixed point offset from axes bottom → no overlap regardless of sub_h
    ax.annotate(caption, xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -_CAP_OFF_PT), textcoords="offset points",
                ha="center", va="top", fontsize=_FS_CAPTION, fontweight="bold",
                annotation_clip=False)


def make_grid(idx_list, split, token_ids_all, sample_token_all, sfwd, slat,
              codebook, show_substeps, title, alternate=False):
    """Render all idx as a uniform-size grid.

    Layout:  square-ish (minimise |nrows - ncols|).
    Canvas:  sized to data extent so all subplots have equal aspect with no whitespace.
    Colorbar: horizontal, centred below title.
    """
    import math

    n = len(idx_list)

    # Square-ish layout: prefer landscape on tie
    best_ncols, best_score = 1, float("inf")
    for nc in range(1, n + 1):
        nr   = math.ceil(n / nc)
        score = abs(nr - nc) * 10 + (1 if nc >= nr else 0)
        if score < best_score:
            best_score, best_ncols = score, nc
    ncols = best_ncols
    nrows = math.ceil(n / ncols)

    # Pre-decode all frames
    frames = []
    for idx in idx_list:
        pos, head, subs, bin_ids = decode_frame(token_ids_all[idx], codebook)
        frames.append((idx, str(sample_token_all[idx]), pos, head, subs, bin_ids))

    # Global data extent (trajectory + last-substep bbox corners)
    xs, ys = [], []
    for _, _, pos, _, subs, _ in frames:
        xs.extend(pos[:, 0].tolist())
        ys.extend(pos[:, 1].tolist())
        for tok_subs in subs:
            xs.extend(tok_subs[-1][:, 0].tolist())
            ys.extend(tok_subs[-1][:, 1].tolist())
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    px = max(2.0, (xmax - xmin) * 0.08)
    py = max(2.0, (ymax - ymin) * 0.18)
    xlim = (xmin - px, xmax + px)
    ylim = (ymin - py, ymax + py)
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    # Subplot size: equal aspect (sub_w/sub_h = dx/dy), minimum 3 in wide
    scale   = max(3.0, dx * 0.10) / dx   # in/m
    sub_w   = dx * scale
    sub_h   = dy * scale

    # Layout margins — all from font metrics
    L    = _LEFT_W          # ylabel + yticks of first column
    R    = 0.15             # small right padding (no colorbar)
    BOT  = _BELOW_H         # xticks + xlabel + caption of bottom row
    TOP  = _TOP_H           # title + padding
    hgap = _BELOW_H + 0.14  # between rows
    wgap = _LEFT_W  + 0.12  # between cols

    fig_w = L + ncols * sub_w + (ncols - 1) * wgap + R
    fig_h = TOP + nrows * sub_h + (nrows - 1) * hgap + BOT

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#ffffff")

    gs = fig.add_gridspec(
        nrows, ncols,
        left   = L / fig_w,
        right  = 1 - R / fig_w,
        bottom = BOT / fig_h,
        top    = 1 - TOP / fig_h,
        hspace = hgap / sub_h,
        wspace = wgap / sub_w,
    )

    for pos_i, (idx, orig, pos, head, subs, bin_ids) in enumerate(frames):
        r, c  = divmod(pos_i, ncols)
        ax    = fig.add_subplot(gs[r, c])
        caption = f"fwd={sfwd[idx]:.2f} m/step · lat={slat[idx]:.2f} m/step\n{orig}"
        _draw_subplot(ax, pos, head, subs, bin_ids, show_substeps, caption,
                      xlim=xlim, ylim=ylim, alternate=alternate)

    for pos_i in range(n, nrows * ncols):
        r, c = divmod(pos_i, ncols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    # Title: two-line, centred over grid
    title_y = 1 - _ph(_FS_TITLE) * 0.55 / fig_h
    fig.text(0.5, title_y, title,
             ha="center", va="center", fontsize=_FS_TITLE, fontweight="bold",
             linespacing=1.4)

    return fig


def plot_single(idx, split, token_ids_all, sample_token_all, sfwd, slat,
                codebook, show_substeps, alternate=False):
    """Render one frame as a standalone figure, canvas sized to data."""
    ids_row = token_ids_all[idx]
    orig    = str(sample_token_all[idx])
    pos, head, subs, bin_ids = decode_frame(ids_row, codebook)

    # Data extent
    xs = list(pos[:, 0]) + [v for ts in subs for v in ts[-1][:, 0]]
    ys = list(pos[:, 1]) + [v for ts in subs for v in ts[-1][:, 1]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    px = max(2.0, (xmax - xmin) * 0.08)
    py = max(2.0, (ymax - ymin) * 0.18)
    xlim = (xmin - px, xmax + px)
    ylim = (ymin - py, ymax + py)
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

    scale = max(3.5, dx * 0.10) / dx
    sub_w, sub_h = dx * scale, dy * scale

    L    = _LEFT_W
    R    = _LEFT_W           # symmetric right padding (no colorbar)
    TOP  = _ph(_FS_TITLE) + 0.14
    BOT  = _BELOW_H
    fig_w = L + sub_w + R
    fig_h = TOP + sub_h + BOT

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#ffffff")

    ax = fig.add_axes([L/fig_w, BOT/fig_h, sub_w/fig_w, sub_h/fig_h])
    caption = f"fwd={sfwd[idx]:.2f} m/step · lat={slat[idx]:.2f} m/step\n{orig}"
    _draw_subplot(ax, pos, head, subs, bin_ids, show_substeps, caption,
                  xlim=xlim, ylim=ylim, alternate=alternate)

    title_y = 1 - _ph(_FS_TITLE) * 0.55 / fig_h
    fig.text(0.5, title_y,
             f"AutoVLA Decoded Trajectories\nnuScenes {split} · Δt=0.5 s",
             ha="center", va="center", fontsize=_FS_TITLE, fontweight="bold",
             linespacing=1.4)

    return fig, pos, head, bin_ids


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Decode & visualize AutoVLA action tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--speed", choices=["high", "mid", "low"],
                   help="Auto-select frames by speed level → grid output")
    g.add_argument("--idx",
                   help="Single int (e.g. 42) → standalone plot; "
                        "comma-separated (e.g. 0,42,6974) → grid output")

    p.add_argument("--split",    default="train", choices=["train", "val"])
    p.add_argument("--lateral",  default="all",
                   choices=["left", "straight", "right", "all"],
                   help="Lateral direction filter (only with --speed, default: all)")
    p.add_argument("--n",        type=int, default=6,
                   help="Number of frames to select (only with --speed, default: 6)")
    p.add_argument("--show-substeps", action="store_true",
                   help="Draw the 6 sub-step bounding boxes per token")
    p.add_argument("--alternate", action="store_true",
                   help="Alternate token-id labels top-right / bottom-right")
    p.add_argument("--output",   default=None,
                   help="Save path (default: show interactively)")
    p.add_argument("--h5", 
                   default=None,
                   help="Path to h5 file")
    p.add_argument("--codebook",
                   default="third_party/AutoVLA/codebook_cache/agent_vocab.pkl",
                   help="Path to codebook pkl")
    args = p.parse_args()
    if args.h5 is None:
        args.h5 = f"outputs/autovla/embeddings/{args.split}_action_embeddings.h5"
    return args


def main():
    args = parse_args()

    with h5py.File(args.h5, 'r') as f:
        token_ids_all    = f["token_ids"][:]             # (S, T) int32
        sample_token_all = f["sample_token"][:].astype(str)  # (S,) str

    codebook = load_codebook(Path(args.codebook))

    # ── single-frame path ──────────────────────────────────────────────────────
    if args.idx is not None and "," not in args.idx:
        idx = int(args.idx)
        if not (0 <= idx < len(token_ids_all)):
            raise SystemExit(f"--idx {idx} out of range [0, {len(token_ids_all)})")
        sfwd, slat = _speed_stats(token_ids_all, codebook)
        print(f"Single frame  id={sample_token_all[idx]}"
              f"  sfwd={sfwd[idx]:.2f}  slat={slat[idx]:.2f}")
        fig, pos, head, bin_ids = plot_single(
            idx, args.split, token_ids_all, sample_token_all,
            sfwd, slat, codebook, args.show_substeps, alternate=args.alternate,
        )
        print(f"\nDecoded trajectory ({len(pos)} waypoints):")
        for i, (p, h) in enumerate(zip(pos, np.degrees(head))):
            print(f"  t={i:2d}  x={p[0]:7.3f}  y={p[1]:7.3f}  heading={h:6.1f}°")

    # ── grid path ─────────────────────────────────────────────────────────────
    else:
        if args.speed:
            idx_list, sfwd, slat = select_idx(
                token_ids_all, codebook,
                speed=args.speed, lateral=args.lateral, n=args.n,
            )
            title = (
                f"AutoVLA Decoded Trajectories\n"
                f"nuScenes {args.split} · speed={args.speed} · lateral={args.lateral}"
                f" · Δt=0.5 s"
            )
        else:
            idx_list = [int(x) for x in args.idx.split(",")]
            sfwd, slat = _speed_stats(token_ids_all, codebook)
            title = (
                f"AutoVLA Decoded Trajectories\n"
                f"nuScenes {args.split} · idx={args.idx} · Δt=0.5 s"
            )

        print(f"Grid {len(idx_list)} samples:")
        for i in idx_list:
            print(f"  token={sample_token_all[i]}"
                  f"  fwd={sfwd[i]:.2f} m/step  lat={slat[i]:.2f} m/step")
        fig = make_grid(idx_list, args.split, token_ids_all, sample_token_all,
                        sfwd, slat, codebook, args.show_substeps, title,
                        alternate=args.alternate)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor="#ffffff")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

