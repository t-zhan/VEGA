"""
Codebook BEV Visualization

3×3 grid:
  rows    — agent type: veh (vehicle), ped (pedestrian), cyc (cyclist)
  columns — speed tier: high, mid, low  (split by per-agent p33/p67)

Each subplot draws 6 dashed bounding boxes (t=1..6) with solid corner markers.
Colours progress from light to dark as time advances.

Usage:
    python src/analysis/autovla/codebook_bev.py \
        --pkl  third_party/AutoVLA/codebook_cache/agent_vocab.pkl \
        --out  outputs/autovla/viz/codebook_bev.png \
        --seed 42
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import colormaps
from matplotlib.patches import FancyArrowPatch
import matplotlib.legend_handler as lh
import numpy as np


class _HandlerAnnotateArrow(lh.HandlerBase):
    """Render a -|> arrow (matching ax.annotate style) inside the legend box."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        arrow = FancyArrowPatch(
            posA=(xdescent, height / 2),
            posB=(xdescent + width, height / 2),
            arrowstyle="-|>",
            color="red",
            lw=1.5,
            mutation_scale=8,
            transform=trans,
        )
        return [arrow]


# ── speed thresholds (6-step endpoint displacement, metres) ──────────────────
# derived from p33/p67 of each agent's distribution
SPEED_THRESHOLDS = {
    "veh": (2.33, 4.44),   # low < 2.33, mid 2.33-4.44, high > 4.44
    "ped": (0.20, 0.52),
    "cyc": (1.08, 2.38),
}

AGENT_LABELS = {"veh": "Vehicle", "ped": "Pedestrian", "cyc": "Cyclist"}
SPEED_LABELS  = ["High Speed", "Mid Speed", "Low Speed"]
AGENT_ORDER   = ["veh", "ped", "cyc"]

# colour maps per agent row
CMAPS = {"veh": "Blues", "ped": "Greens", "cyc": "Oranges"}


ACTION_START_ID = 151665   # offset between codebook index and actual token id


def select_entry(cb: np.ndarray, tier: str, thresholds: tuple, rng: np.random.Generator) -> int:
    """Return index of a randomly chosen entry in the requested speed tier."""
    centers = cb.mean(axis=2)           # (2048, 6, 2)
    speeds  = np.linalg.norm(centers[:, -1], axis=1)   # (2048,)
    lo, hi  = thresholds
    if tier == "high":
        mask = speeds > hi
    elif tier == "mid":
        mask = (speeds >= lo) & (speeds <= hi)
    else:  # low
        mask = speeds < lo
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise RuntimeError(f"No entries found for tier={tier}, thresholds={thresholds}")
    return int(rng.choice(indices))


def draw_entry(ax: plt.Axes, entry: np.ndarray, cmap_name: str) -> None:
    """Draw 6 dashed bboxes with solid corner dots onto ax.

    entry shape: (6, 4, 2)  —  timesteps × corners × (x,y)
    Corner order from cal_polygon_contour: left-front, right-front, right-back, left-back
    """
    n_steps = entry.shape[0]
    cmap    = colormaps[cmap_name]
    norm    = Normalize(vmin=-1, vmax=n_steps - 1)   # vmin=-1 to avoid pure white at t=0

    for t in range(n_steps):
        corners = entry[t]                 # (4, 2)
        colour  = cmap(norm(t))

        # closed dashed polygon
        poly_xy = np.vstack([corners, corners[0]])   # close the loop
        ax.plot(poly_xy[:, 0], poly_xy[:, 1],
                linestyle="--", linewidth=1.2, color=colour, alpha=0.9)

        # solid corner markers
        ax.scatter(corners[:, 0], corners[:, 1],
                   s=18, color=colour, zorder=5, edgecolors="none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",  default="third_party/AutoVLA/codebook_cache/agent_vocab.pkl")
    parser.add_argument("--out",  default="outputs/autovla/viz/codebook_bev.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        vocab = pickle.load(f)["token_all"]   # {"veh": ndarray, "ped": ..., "cyc": ...}

    rng = np.random.default_rng(seed=args.seed)
    tiers = ["high", "mid", "low"]

    fig, axes = plt.subplots(3, 3, figsize=(11, 10),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.42})
    fig.suptitle("AutoVLA Codebook BEV — agent_vocab.pkl", fontsize=14, fontweight="bold", y=1.02)

    entry_indices = []   # list[list[int]], shape (3, 3) — row-major (agent × tier)

    for row_idx, agent in enumerate(AGENT_ORDER):
        cb     = vocab[agent]           # (2048, 6, 4, 2)
        thresh = SPEED_THRESHOLDS[agent]
        cmap   = CMAPS[agent]
        row_indices = []

        for col_idx, tier in enumerate(tiers):
            ax  = axes[row_idx][col_idx]
            idx = select_entry(cb, tier, thresh, rng)
            row_indices.append(idx)

            draw_entry(ax, cb[idx], cmap)

            # arrow from origin to last-step center; red star at destination
            last_center = cb[idx, -1].mean(axis=0)   # (2,) mean of 4 corners at t=6
            ax.annotate("", xy=(last_center[0], last_center[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="red",
                                        lw=1.5, mutation_scale=14), zorder=9)
            ax.scatter([last_center[0]], [last_center[1]],
                       marker="*", s=80, color="red", zorder=10)

            # adjustable="datalim" keeps every axes box the same physical size;
            # only the visible data range is adjusted to maintain equal aspect.
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.tick_params(labelsize=12)

            # token id below subplot (combined with x-axis label); y axis label
            ax.set_title("")
            ax.set_xlabel(f"x (m)\ntoken[{ACTION_START_ID + idx}]", fontsize=14, color="#444444", labelpad=3)
            ax.set_ylabel("y (m)", fontsize=14, color="#444444", labelpad=0)

        entry_indices.append(row_indices)

    LEGEND_ANCHOR_Y = 0.990   # legend top anchor — just below suptitle at y=1.02

    legend_elements = [
        mpatches.Patch(color=colormaps["Greys"](0.3), label="t=1 (earliest)"),
        mpatches.Patch(color=colormaps["Greys"](0.8), label="t=6 (latest)"),
        plt.Line2D([0], [0], linestyle="--", color="gray", label="bbox (dashed)"),
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="none",
                   markersize=5, label="corners (solid)"),
        mpatches.Patch(color="red", label="displacement"),
        plt.Line2D([0], [0], marker="*", color="red", linestyle="none",
                   markersize=7, label="last step center"),
    ]

    _arrow_handler = {legend_elements[4]: _HandlerAnnotateArrow()}

    # ── Pass 1: rough layout → measure title/legend bounding boxes in pixels ───
    plt.subplots_adjust(left=0.20, bottom=0.17, right=0.97, top=0.92)
    legend = fig.legend(handles=legend_elements, loc="upper center", ncol=6,
                        fontsize=8, bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
                        handler_map=_arrow_handler)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.get_figheight() * fig.get_dpi()
    title_bb = fig._suptitle.get_window_extent(renderer)
    leg_bb   = legend.get_window_extent(renderer)

    # gap: title-bottom → legend-top; apply 2× same gap: legend-bottom → subplots-top
    gap_px  = title_bb.y0 - leg_bb.y1
    new_top = (leg_bb.y0 - 2.0 * gap_px) / fig_h_px

    # ── Pass 2: apply corrected top, then place row/col labels ────────────────
    # Do NOT touch fig.texts — suptitle lives there and must be preserved.
    legend.remove()
    plt.subplots_adjust(top=max(0.85, new_top))
    fig.canvas.draw()

    # ── Row labels ────────────────────────────────────────────────────────────
    ROW_GAP = 0.08   # push row labels further left to avoid y-axis overlap
    for row_idx, agent in enumerate(AGENT_ORDER):
        pos = axes[row_idx][0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(pos.x0 - ROW_GAP, y_center, AGENT_LABELS[agent],
                 ha="right", va="center", rotation=90,
                 fontsize=12, fontweight="bold")

    # ── Column labels ─────────────────────────────────────────────────────────
    bottom_row_y = axes[2][0].get_position().y0
    col_label_y  = bottom_row_y - 0.085
    for col_idx, speed_label in enumerate(SPEED_LABELS):
        pos      = axes[2][col_idx].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, col_label_y, speed_label,
                 ha="center", va="top",
                 fontsize=12, fontweight="bold")

    # ── Align all y-axis labels to the same x position ────────────────────────
    fig.align_ylabels(axes)

    # ── Final legend ──────────────────────────────────────────────────────────
    fig.legend(handles=legend_elements, loc="upper center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
               handler_map=_arrow_handler)

    # Build output path with token-id suffix:
    # codebook_bev_{veh_h}-{veh_m}-{veh_l}_{ped_h}-{ped_m}-{ped_l}_{cyc_h}-{cyc_m}-{cyc_l}.png
    idx_suffix = "_".join("-".join(str(ACTION_START_ID + i) for i in row) for row in entry_indices)
    out_path = Path(args.out)
    out_path = out_path.parent / f"{out_path.stem}_{idx_suffix}_v8{out_path.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
