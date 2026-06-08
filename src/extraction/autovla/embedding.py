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
    tokenizer=None,
    T_text: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract last-layer hidden states at action token positions via teacher forcing.

    Calls vlm(**inputs, output_hidden_states=True) directly, bypassing
    AutoVLA.forward() which does not expose output_hidden_states.

    Args:
        vlm: The inner Qwen2.5-VL model (AutoVLA.vlm).
        dataloader: Yields batches compatible with Qwen2.5-VL forward signature.
                    Each batch must contain input_ids and labels (ground truth).
        action_start_id: Vocabulary index of <action_0>.
        device: Device to run inference on.
        tokenizer: Tokenizer for encoding "</think>" to locate CoT final text tokens.
        T_text: Number of text tokens before </think> to extract (default 10).

    Returns:
        token_ids      : ndarray (S, T_action)             — action token ids per sample
        hidden_vecs    : ndarray (S, T_action, hidden_dim) — last-layer hidden states
        text_token_ids : ndarray (S, T_text)               — last T_text text tokens before </think>
        text_hidden    : ndarray (S, T_text, hidden_dim)   — hidden states for those tokens
        sample_indices : ndarray (S,)                      — dataset indices for kept samples
    """
    if tokenizer is not None:
        think_end_ids = tokenizer.encode("</think>\n", add_special_tokens=False)
        think_end_t = torch.tensor(think_end_ids, device=device)
        L = len(think_end_ids)
    else:
        think_end_t = None

    vlm = vlm.to(device)
    vlm.eval()

    all_token_ids: list[np.ndarray] = []
    all_hidden: list[np.ndarray] = []
    all_text_tids: list[np.ndarray] = []
    all_text_hidden: list[np.ndarray] = []
    sample_indices: list[int] = []
    global_idx = 0

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
            T_action = action_mask.sum(dim=1)[0].item()  # action tokens per sample (fixed)

            # (B, T_action) and (B, T_action, D) — keep sample dimension intact
            action_tids = labels[action_mask].view(B, T_action)
            action_hidden = last_hidden[action_mask].view(B, T_action, D)

            # Extract last T_text text tokens before </think>
            if think_end_t is not None:
                text_tids_batch = []
                text_hidden_batch = []
                keep_indices = []
                for i in range(B):
                    first_act = (labels[i] >= action_start_id).nonzero()[0].item()
                    # search backwards for </think>
                    think_end_pos = None
                    for pos in range(first_act - L, -1, -1):
                        if torch.equal(labels[i, pos:pos + L], think_end_t):
                            think_end_pos = pos
                            break
                    if think_end_pos is None:
                        continue  # skip samples without </think>
                    keep_indices.append(i)
                    start = think_end_pos - T_text
                    text_tids_batch.append(labels[i, start:think_end_pos])
                    text_hidden_batch.append(last_hidden[i, start:think_end_pos])

                if keep_indices:
                    keep = torch.tensor(keep_indices, device=labels.device)
                    all_token_ids.append(action_tids[keep].cpu().numpy())
                    all_hidden.append(action_hidden[keep].float().cpu().numpy())
                    all_text_tids.append(torch.stack(text_tids_batch).cpu().numpy())
                    all_text_hidden.append(torch.stack(text_hidden_batch).float().cpu().numpy())
                    for i in keep_indices:
                        sample_indices.append(global_idx + i)
                # else: entire batch skipped — no </think> in any sample
            else:
                all_token_ids.append(action_tids.cpu().numpy())
                all_hidden.append(action_hidden.float().cpu().numpy())
                for i in range(B):
                    sample_indices.append(global_idx + i)
            global_idx += B

    result = [
        np.concatenate(all_token_ids, axis=0),   # (S, T_action)
        np.concatenate(all_hidden, axis=0),       # (S, T_action, D)
    ]
    if think_end_t is not None:
        result.append(np.concatenate(all_text_tids, axis=0))    # (S, T_text)
        result.append(np.concatenate(all_text_hidden, axis=0))  # (S, T_text, D)
    else:
        result.append(None)
        result.append(None)
    result.append(np.array(sample_indices, dtype=int))  # (S,) dataset indices
    return tuple(result)
