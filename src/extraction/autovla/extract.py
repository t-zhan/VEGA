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
    first_embed     (2048, hidden_dim)         first-layer embed_tokens.weight rows for action tokens
    token_ids       (S, T_action)              action token ids per sample
    last_hidden     (S, T_action, hidden_dim)  last-layer hidden state per sample (gzip compressed)
    sample_token    (S,)                       nuScenes sample token (32-char hex) for each sample
    text_token_ids  (S, T_text)               last T_text text tokens before </think> in CoT
    text_hidden     (S, T_text, hidden_dim)    hidden states for those text tokens
    text_first_embed (vocab_size_text, hidden_dim)  first-layer embed_tokens.weight rows for all text tokens
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Allow extraction/ to import from visualize/
_SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from visualize.autovla.decode_trajectory import select_idx, load_codebook


def load_filtered_embeddings(
    h5_path,
    speed=None,
    lateral="all",
    n=6,
    sample_tokens=None,
    codebook_path="third_party/AutoVLA/codebook_cache/agent_vocab.pkl",
):
    """Load and optionally filter action embeddings from an HDF5 file."""
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        if sample_tokens is not None:
            all_tokens = f["sample_token"][:].astype(str)
            sel = np.sort(np.where(np.isin(all_tokens, sample_tokens))[0])
        elif speed is not None:
            token_ids_all = f["token_ids"][:]
            codebook = load_codebook(Path(codebook_path))
            sel, sfwd, slat = select_idx(
                token_ids_all, codebook, speed=speed, lateral=lateral, n=n,
            )
            sel = np.sort(np.asarray(sel, dtype=int))
        else:
            sel = slice(None)

        result = {
            "token_ids":    f["token_ids"][sel],
            "last_hidden":  f["last_hidden"][sel],
            "first_embed":  f["first_embed"][:],
            "sample_token": f["sample_token"][sel],
        }
        if "text_token_ids" in f:
            result["text_token_ids"]  = f["text_token_ids"][sel]
            result["text_hidden"]     = f["text_hidden"][sel]
            if "text_first_embed" in f:
                result["text_first_embed"] = f["text_first_embed"][:]  # static, no sample filtering needed
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Extract AutoVLA action token embeddings")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--mode", default="teacher-forcing",
                        choices=["teacher-forcing", "autoregressive"],
                        help="Extraction mode (default: teacher-forcing)")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Dataset split")
    parser.add_argument("--output", default="outputs/action_embeddings.h5",
                        help="Output .h5 file path (default: outputs/action_embeddings.h5)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for hidden extraction")
    parser.add_argument("--start", type=int, default=None, help="Start sample index (inclusive, default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End sample index (exclusive, default: len(dataset))")
    parser.add_argument("--T-text", type=int, default=10, help="Number of text tokens before </think> to extract (default: 10)")
    return parser.parse_args()


def _collect_tokens(dataset, indices):
    """Return list of sample_token strings at given dataset indices."""
    if hasattr(dataset, 'dataset'):
        # Subset: map through dataset.indices
        full_indices = dataset.indices
        scenes = [dataset.dataset.scenes[full_indices[i]] for i in indices]
    else:
        scenes = [dataset.scenes[i] for i in indices]
    return [Path(scene[0] if isinstance(scene, tuple) else scene).stem for scene in scenes]


def main():
    args = parse_args()

    from loaders import load_autovla, load_dataloader, load_prompt_dataloader
    from embedding import extract_static, extract_hidden, extract_autoregressive

    print(f"[1/4] Loading model from checkpoint: {args.ckpt}")
    model = load_autovla(args.config, args.ckpt, device=args.device)
    action_start_id = model.action_start_id

    print(f"[2/4] Extracting first-layer action embeddings")
    first_embed = extract_static(model.vlm, action_start_id=action_start_id)
    print(f"      shape: {first_embed.shape}")

    print(f"[3/4] Building dataloader (split={args.split})")
    load_fn = load_prompt_dataloader if args.mode == "autoregressive" else load_dataloader
    dataloader = load_fn(args.config, split=args.split, batch_size=args.batch_size,
                         start=args.start, end=args.end)
    print(f"      {len(dataloader.dataset)} samples")

    if args.mode == "autoregressive":
        print(f"[4/4] Extracting last-layer hidden states (autoregressive generation)")
        token_ids, last_hidden, text_tids, text_hidden, kept_idx = extract_autoregressive(
            model, dataloader, action_start_id=action_start_id, device=args.device,
            tokenizer=model.processor.tokenizer, T_text=args.T_text,
        )
    else:
        print(f"[4/4] Extracting last-layer hidden states (teacher forcing)")
        token_ids, last_hidden, text_tids, text_hidden, kept_idx = extract_hidden(
            model.vlm, dataloader, action_start_id=action_start_id, device=args.device,
            tokenizer=model.processor.tokenizer, T_text=args.T_text,
        )
    print(f"      extracted {last_hidden.shape[0]} samples, {last_hidden.shape[1]} action tokens each")
    if text_tids is not None:
        print(f"      text tokens: {text_tids.shape[1]} per sample, hidden shape: {text_hidden.shape}")

    sample_tokens = _collect_tokens(dataloader.dataset, kept_idx)

    # Extract full text token embeddings (static, same as action first_embed)
    text_first = model.vlm.model.embed_tokens.weight[
        :action_start_id
    ].detach().float().cpu().numpy()                                         # (vocab_size_text, hidden_dim)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["generation_mode"] = args.mode
        f.create_dataset("first_embed",  data=first_embed, dtype="float32")
        f.create_dataset("token_ids",    data=token_ids,   dtype="int32")
        f.create_dataset("last_hidden",  data=last_hidden, dtype="float32",
                         compression="gzip", compression_opts=4)
        f.create_dataset("sample_token", data=np.array(sample_tokens, dtype=h5py.string_dtype()))
        f.create_dataset("text_first_embed", data=text_first, dtype="float32")
        if text_tids is not None:
            f.create_dataset("text_token_ids", data=text_tids, dtype="int32")
            f.create_dataset("text_hidden",    data=text_hidden, dtype="float32")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
