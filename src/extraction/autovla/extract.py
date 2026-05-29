"""
CLI script: extract action token embeddings (first layer + last layer) from AutoVLA.

Usage:
    python src/extraction/run_extraction.py \
        --config  third_party/AutoVLA/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml \
        --ckpt    third_party/AutoVLA/runs/sft/2026-05-24_19-17-19/epoch=0-loss=0.9887.ckpt \
        --split   val \
        --output  outputs/action_embeddings.h5 \
        --device  cuda

Output .h5 fields:
    first_embed  (2048, hidden_dim)         first-layer embed_tokens.weight rows for action tokens
    token_ids    (S, T)                     action token ids per sample
    last_hidden  (S, T, hidden_dim)         last-layer hidden state per sample (gzip compressed)
    sample_token (S,)                       nuScenes sample token (32-char hex) for each sample
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from loaders import load_autovla, load_dataloader
from embedding import extract_static, extract_hidden


def parse_args():
    parser = argparse.ArgumentParser(description="Extract AutoVLA action token embeddings")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Dataset split to run teacher-forcing on (default: val)")
    parser.add_argument("--output", default="outputs/action_embeddings.h5",
                        help="Output .h5 file path (default: outputs/action_embeddings.h5)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for hidden extraction")
    parser.add_argument("--start", type=int, default=None, help="Start sample index (inclusive, default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End sample index (exclusive, default: len(dataset))")
    return parser.parse_args()


def _collect_tokens(dataset, start, n):
    """Return list of sample_token strings for n consecutive samples starting at start."""
    # dataset may be a torch Subset when --start/--end is used
    if hasattr(dataset, 'dataset'):
        # Subset: dataset.indices already contains the slice
        scenes = [dataset.dataset.scenes[i] for i in dataset.indices]
    else:
        scenes = dataset.scenes[start: start + n]
    return [Path(scene_path).stem for scene_path, _ in scenes]


def main():
    args = parse_args()

    print(f"[1/4] Loading model from checkpoint: {args.ckpt}")
    model = load_autovla(args.config, args.ckpt, device=args.device)
    action_start_id = model.action_start_id

    print(f"[2/4] Extracting first-layer (static) action embeddings")
    static = extract_static(model.vlm, action_start_id=action_start_id)
    print(f"      shape: {static.shape}")

    print(f"[3/4] Building dataloader (split={args.split})")
    dataloader = load_dataloader(args.config, split=args.split, batch_size=args.batch_size,
                                 start=args.start, end=args.end)
    print(f"      {len(dataloader.dataset)} samples")

    print(f"[4/4] Extracting last-layer hidden states (teacher forcing)")
    token_ids, hidden_vecs = extract_hidden(
        model.vlm, dataloader, action_start_id=action_start_id, device=args.device
    )
    print(f"      extracted {hidden_vecs.shape[0]} samples, {hidden_vecs.shape[1]} tokens each, hidden shape: {hidden_vecs.shape}")

    start = args.start or 0
    sample_tokens = _collect_tokens(dataloader.dataset, start, hidden_vecs.shape[0])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("first_embed",  data=static,      dtype="float32")
        f.create_dataset("token_ids",    data=token_ids,   dtype="int32")
        f.create_dataset("last_hidden",  data=hidden_vecs, dtype="float32",
                         compression="gzip", compression_opts=4)
        f.create_dataset("sample_token", data=np.array(sample_tokens, dtype=h5py.string_dtype()))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
