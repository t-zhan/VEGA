"""
Extract action token embeddings from AutoVLA.

Two extraction modes:
  - extract_static : read embed_tokens.weight rows for action tokens (first layer, no inference needed)
  - extract_hidden : teacher-forcing forward pass, capture last Transformer layer output at
                     action token positions (last layer)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_static(
    vlm: torch.nn.Module,
    action_start_id: int = 151665,
    n_action: int = 2048,
) -> np.ndarray:
    """Extract the first-layer (input) embedding vectors for all action tokens.

    Since lm_head.weight is tied to embed_tokens.weight, these vectors are also
    the output logit direction for each action token.

    Args:
        vlm: The inner Qwen2.5-VL model (AutoVLA.vlm).
        action_start_id: Vocabulary index of <action_0>.
        n_action: Total number of action tokens (codebook size).

    Returns:
        ndarray of shape (n_action, hidden_dim), float32.
    """
    weight = vlm.model.embed_tokens.weight  # (vocab_size, hidden_dim)
    action_embeddings = weight[action_start_id : action_start_id + n_action]
    return action_embeddings.detach().float().cpu().numpy()


def extract_hidden(
    vlm: torch.nn.Module,
    dataloader: DataLoader,
    action_start_id: int = 151665,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract last-layer hidden states at action token positions via teacher forcing.

    Calls vlm(**inputs, output_hidden_states=True) directly, bypassing
    AutoVLA.forward() which does not expose output_hidden_states.

    Args:
        vlm: The inner Qwen2.5-VL model (AutoVLA.vlm).
        dataloader: Yields batches compatible with Qwen2.5-VL forward signature.
                    Each batch must contain input_ids and labels (ground truth).
        action_start_id: Vocabulary index of <action_0>.
        device: Device to run inference on.

    Returns:
        token_ids  : ndarray (S, T)             — action token ids per sample
        hidden_vecs: ndarray (S, T, hidden_dim) — last-layer hidden states per sample
    """
    vlm = vlm.to(device)
    vlm.eval()

    all_token_ids: list[np.ndarray] = []
    all_hidden: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting hidden states"):
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ("gt_trajectory", "gt_action", "has_cot")
            }
            labels = inputs.pop("labels")  # (B, T)

            outputs = vlm(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (B, T, hidden_dim)

            # Identify action token positions from ground-truth labels
            action_mask = labels >= action_start_id  # (B, T)
            B, D = labels.shape[0], last_hidden.shape[-1]
            T = action_mask.sum(dim=1)[0].item()  # tokens per sample (fixed)

            # (B, T) and (B, T, D) — keep sample dimension intact
            all_token_ids.append(labels[action_mask].view(B, T).cpu().numpy())
            all_hidden.append(last_hidden[action_mask].view(B, T, D).float().cpu().numpy())

    return (
        np.concatenate(all_token_ids, axis=0),   # (S, T)
        np.concatenate(all_hidden, axis=0),       # (S, T, D)
    )
