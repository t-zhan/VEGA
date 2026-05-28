"""
CLI script: extract action token embeddings (first layer + last layer) from AutoVLA.

Usage:
    python src/extraction/run_extraction.py \
        --config  third_party/AutoVLA/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml \
        --ckpt    third_party/AutoVLA/runs/sft/2026-05-24_19-17-19/epoch=0-loss=0.9887.ckpt \
        --split   val \
        --output  outputs/action_embeddings.npz \
        --device  cuda

Output .npz fields:
    first_embed (2048, hidden_dim)      first-layer embed_tokens.weight rows for action tokens
    token_ids   (S, T)                  action token ids per sample
    last_hidden (S, T, hidden_dim)      last-layer hidden state per sample
    sample_idx  (S,)                    original dataset index for each sample
"""

import argparse
from pathlib import Path

import numpy as np

from loaders import load_autovla, load_dataloader
from embedding import extract_static, extract_hidden


def parse_args():
    parser = argparse.ArgumentParser(description="Extract AutoVLA action token embeddings")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Dataset split to run teacher-forcing on (default: val)")
    parser.add_argument("--output", default="outputs/action_embeddings.npz",
                        help="Output .npz file path (default: outputs/action_embeddings.npz)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for hidden extraction")
    parser.add_argument("--start", type=int, default=None, help="Start sample index (inclusive, default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End sample index (exclusive, default: len(dataset))")
    return parser.parse_args()


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
    sample_idx = np.arange(start, start + hidden_vecs.shape[0])  # (S,) original dataset indices

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        first_embed=static,
        token_ids=token_ids,
        last_hidden=hidden_vecs,
        sample_idx=sample_idx,
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
