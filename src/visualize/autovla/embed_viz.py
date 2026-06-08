"""
Dimensionality-reduction visualization of action token embeddings.

Usage:
    python src/visualize/autovla/embed_viz.py \
        --h5     outputs/autovla/embeddings/train_action_embeddings.h5 \
        --tokens ae291f52e7734887a310bfb843a0119b,ae374f827d2d45c7a4ed054b5aab7370 \
        --method umap
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from extraction.autovla.extract import load_filtered_embeddings


def reduce(arr, method="umap", ndim=2):
    """Reduce *arr* (N, D).  method: 'umap' | 'pca' | 'tsne'."""
    if method == "pca":
        return PCA(n_components=ndim, random_state=42).fit_transform(arr)
    if method == "tsne":
        return TSNE(n_components=ndim, random_state=42, perplexity=min(30, len(arr) - 1)).fit_transform(arr)
    return umap.UMAP(n_components=ndim, random_state=42, n_jobs=1).fit_transform(arr)


def parse_args():
    p = argparse.ArgumentParser(description="Dimensionality-reduction viz of action token embeddings")
    p.add_argument("--h5", required=True, help="Path to action_embeddings.h5")
    p.add_argument("--tokens", default=None,
                   help="Comma-separated nuScenes sample tokens")
    p.add_argument("--speed", default=None, choices=["high", "mid", "low"])
    p.add_argument("--lateral", default="all", choices=["left", "straight", "right", "all"])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--method", default="umap", choices=["umap", "pca", "tsne"])
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3],
                   help="Output dimensions (default: 2)")
    p.add_argument("--codebook",
                   default="third_party/AutoVLA/codebook_cache/agent_vocab.pkl")
    p.add_argument("--output", default="outputs/autovla/viz/embed_viz.png")
    return p.parse_args()


def main():
    args = parse_args()

    if args.tokens:
        sample_tokens = [t.strip() for t in args.tokens.split(",")]
        data = load_filtered_embeddings(args.h5, sample_tokens=sample_tokens)
    elif args.speed:
        data = load_filtered_embeddings(
            args.h5, speed=args.speed, lateral=args.lateral, n=args.n,
            codebook_path=args.codebook)
        sample_tokens = data["sample_token"].astype(str).tolist()
    else:
        raise SystemExit("Specify --tokens or --speed.")

    first_embed = data["first_embed"]                              # (2048, D)
    token_ids = data["token_ids"]                                  # (S', T_action)

    used_bins = np.unique(token_ids - 151665)
    used_embed = first_embed[used_bins]                            # (K, D)
    used_token_ids = used_bins + 151665                            # (K,) real token IDs

    last_hidden = data["last_hidden"].reshape(-1, first_embed.shape[-1])  # (S*T_action, D)
    last_token_ids = token_ids.reshape(-1)                                  # (S*T_action,)

    has_text = "text_hidden" in data and "text_token_ids" in data
    if has_text:
        text_hidden = data["text_hidden"].reshape(-1, first_embed.shape[-1])  # (S*T_text, D)
        text_tids = data["text_token_ids"].reshape(-1)                        # (S*T_text,)
    has_text_first = has_text and "text_first_embed" in data
    if has_text_first:
        text_first_flat = data["text_first_embed"][:].reshape(-1, first_embed.shape[-1])  # (S*T_text, D)
        _, unique_idx = np.unique(text_tids, return_index=True)  # dedup like action side
        text_first = text_first_flat[unique_idx]                 # (K_text, D)

    # Per-token lateral displacement (positive = left, negative = right)
    from decode_trajectory import load_codebook
    codebook_t = load_codebook(Path(args.codebook))
    token_lat = codebook_t[:, -1].mean(dim=1)[:, 1].numpy()  # (2048,)

    fe_lat = token_lat[used_bins]                     # (K,)
    lh_lat = token_lat[last_token_ids - 151665]       # (S*T_action,)

    fe_colors = fe_lat
    lh_colors = lh_lat

    lo, hi = np.percentile(np.concatenate([fe_lat, lh_lat]), [2, 98])
    norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)

    F = 14 if args.ndim == 3 else 18   # unified font size for 3D, larger for 2D
    is_3d = args.ndim == 3
    if is_3d:
        fig = plt.figure(figsize=(18, 7))
        # proj_type="ortho": perspective projection foreshortens the grid so ticks
        # appear offset from grid lines; orthographic keeps them aligned.
        ax0 = fig.add_axes([0.10, 0.10, 0.26, 0.78], projection="3d", proj_type="ortho")
        ax1 = fig.add_axes([0.40, 0.10, 0.26, 0.78], projection="3d", proj_type="ortho")
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 8))

    import matplotlib.ticker as mticker

    def _snap_3d(ax, arr, nbins=6):
        """Snap each axis range to rounded tick values so ticks, grid lines and
        the box edges all coincide (avoids the 'ticks outside the box' mismatch)."""
        ax.set_box_aspect(None)
        for dim, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            axis = (ax.xaxis, ax.yaxis, ax.zaxis)[dim]
            lo, hi = arr[:, dim].min(), arr[:, dim].max()
            ticks = mticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
            setter(ticks[0], ticks[-1])
            axis.set_major_locator(mticker.FixedLocator(ticks))

    def _draw(ax, arr, colors, title, annot_token_ids, text_arr=None):
        if text_arr is not None:
            combined = np.concatenate([arr, text_arr], axis=0)
            p_all = reduce(combined, args.method, args.ndim)
            p = p_all[:len(arr)]
            tp = p_all[len(arr):]
        else:
            p = reduce(arr, args.method, args.ndim)
            tp = None
        if is_3d:
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=14, alpha=1.0,
                       c=colors, cmap="RdYlBu", norm=norm)
            ax.set_zlabel("Dim 3", fontsize=F-4, labelpad=8)
            _snap_3d(ax, p if tp is None else p_all)
            if tp is not None:
                ax.scatter(tp[:, 0], tp[:, 1], tp[:, 2], s=8, alpha=0.2,
                           c="gray")
        else:
            ax.scatter(p[:, 0], p[:, 1], s=18, alpha=1.0,
                       c=colors, cmap="RdYlBu", norm=norm)
            if tp is not None:
                ax.scatter(tp[:, 0], tp[:, 1], s=10, alpha=0.2,
                           c="gray")
        # for (x, y), tid in zip(p, annot_token_ids):
        #     ax.annotate(f"#{tid}", (x, y), fontsize=9, alpha=0.8,
        #                  xytext=(3, 3), textcoords="offset points")
        if not is_3d:
            ax.set_title(title, fontsize=F, y=-0.17)
        ax.set_xlabel("Dim 1", fontsize=F-4 if is_3d else F, labelpad=12 if is_3d else 4)
        ax.set_ylabel("Dim 2", fontsize=F-4 if is_3d else F, labelpad=12 if is_3d else 4)
        ax.tick_params(labelsize=F - 4 if is_3d else F)

    _draw(ax0, used_embed, fe_colors, "First-layer embed", used_token_ids,
          text_arr=text_first if has_text_first else None)
    _draw(ax1, last_hidden, lh_colors, "Last-layer hidden", last_token_ids,
          text_arr=text_hidden if has_text else None)

    parts = [args.method.upper(), f"{len(sample_tokens)} sample(s)"]
    if args.speed:
        parts.insert(1, f"{args.speed} speed · {args.lateral}")
    # 3-D subplots span x in [0.10, 0.66] → numeric centre 0.38
    sup_x = 0.38 if is_3d else 0.5
    suptitle = fig.suptitle(" · ".join(parts), fontsize=F, fontweight="bold",
                            x=sup_x, y=0.98 if not is_3d else 0.94)

    cbar_y = 0.88 if is_3d else 0.9
    cbar_ax = fig.add_axes([sup_x - 0.15, cbar_y, 0.3, 0.015])
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdYlBu")
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=F-4 if is_3d else F)

    if not is_3d:
        fig.subplots_adjust(wspace=0.25, top=0.82, bottom=0.12)
        title_texts = []
    else:
        # subplot titles placed just below each 3-D axes (centred on rect x-extent)
        title_texts = [
            fig.text(0.23, 0.13, "First-layer embed", ha="center", va="top", fontsize=F-4),
            fig.text(0.53, 0.13, "Last-layer hidden", ha="center", va="top", fontsize=F-4),
        ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: Axes3D.get_tightbbox() ignores auto-placed axis labels, so under
    # bbox_inches="tight" the right subplot's z-label ("Dim 3") and the suptitle
    # would be cropped. Pass them explicitly so the tight bbox includes them.
    extra = [suptitle]
    if is_3d:
        extra += [ax0.zaxis.label, ax1.zaxis.label,
                  ax0.xaxis.label, ax1.xaxis.label,
                  ax0.yaxis.label, ax1.yaxis.label,
                  *title_texts]
    fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=extra)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
