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


def extract_autoregressive(
    vlm: torch.nn.Module,
    dataloader: DataLoader,
    action_start_id: int = 151665,
    device: str = "cuda",
    tokenizer=None,
    T_text: int = 10,
    max_new_tokens: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract last-layer hidden states via autoregressive generation.

    Uses vlm.generate() instead of vlm.forward(). The model generates its own
    CoT reasoning → </think> → action tokens, without ground-truth labels.

    Args:
        vlm: The inner Qwen2.5-VL model (AutoVLA.vlm).
        dataloader: Yields batches. Only input_ids and pixel_values are used;
                    labels are ignored.
        action_start_id: Vocabulary index of <action_0>.
        device: Device to run inference on.
        tokenizer: Tokenizer for encoding "</think>" to locate text/action boundary.
        T_text: Number of text tokens before </think> to extract.
        max_new_tokens: Maximum tokens to generate per sample.

    Returns:
        Same format as extract_hidden:
        token_ids, hidden_vecs, text_token_ids, text_hidden, sample_indices
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
        for batch in tqdm(dataloader, desc="Autoregressive extraction"):
            B = batch["input_ids"].shape[0]

            # Only pass prompt inputs (no labels)
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ("labels", "gt_trajectory", "gt_action", "has_cot")
            }

            outputs = vlm.generate(
                **inputs,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.2,
            )

            # Stack last-layer hidden state per generation step → (B, T_gen, D)
            # Step 0: hs[-1] includes all prefill positions → take last token
            # Steps 1+: hs[-1] is (B, 1, D) → squeeze
            last_hidden_flat = []
            for step_i, hs in enumerate(outputs.hidden_states):
                h = hs[-1]  # last layer
                if step_i == 0 and h.shape[1] > 1:
                    h = h[:, -1:, :]  # take last prefill token only
                last_hidden_flat.append(h.squeeze(1))
            last_hidden = torch.stack(last_hidden_flat, dim=1)
            # outputs.sequences: (B, prefill_len + T_gen)
            gen_ids = outputs.sequences  # full sequence

            T_gen = last_hidden.shape[1]
            D = last_hidden.shape[-1]
            prefill_len = gen_ids.shape[1] - T_gen

            # Identify action token positions in the generated region only
            action_mask = gen_ids[:, prefill_len:] >= action_start_id  # (B, T_gen)

            # Process each sample in batch individually (T_action varies)
            for i in range(B):
                gen_act_pos = action_mask[i].nonzero().squeeze(1)  # (n_act,)
                if len(gen_act_pos) < 10:
                    continue  # drop sample: fewer than 10 generated action tokens

                gen_act_pos = gen_act_pos[:10]
                act_tids = gen_ids[i, prefill_len + gen_act_pos]
                act_hidden = last_hidden[i, gen_act_pos]  # (10, D)

                # Extract text tokens before </think>
                if think_end_t is not None:
                    first_act_pos = prefill_len + gen_act_pos[0].item()
                    think_end_pos = None
                    for pos in range(first_act_pos - L, -1, -1):
                        if torch.equal(gen_ids[i, pos:pos + L], think_end_t):
                            think_end_pos = pos
                            break
                    if think_end_pos is None or think_end_pos - T_text < prefill_len:
                        continue  # drop sample: no </think> or text window not fully generated

                    start_t = think_end_pos - T_text
                    text_tids_i = gen_ids[i, start_t:think_end_pos].unsqueeze(0)  # (1, T_text)

                    text_gen_pos = torch.arange(start_t, think_end_pos, device=device) - prefill_len
                    text_hidden_i = last_hidden[i, text_gen_pos].unsqueeze(0)  # (1, T_text, D)

                    all_token_ids.append(act_tids.unsqueeze(0).cpu().numpy())
                    all_hidden.append(act_hidden.unsqueeze(0).float().cpu().numpy())
                    all_text_tids.append(text_tids_i.cpu().numpy())
                    all_text_hidden.append(text_hidden_i.float().cpu().numpy())
                    sample_indices.append(global_idx + i)
                else:
                    all_token_ids.append(act_tids.unsqueeze(0).cpu().numpy())
                    all_hidden.append(act_hidden.unsqueeze(0).float().cpu().numpy())
                    sample_indices.append(global_idx + i)

            global_idx += B

    result = [
        np.concatenate(all_token_ids, axis=0),   # (S, T_fixed)
        np.concatenate(all_hidden, axis=0),       # (S, T_fixed, D)
    ]
    if think_end_t is not None and all_text_tids:
        result.append(np.concatenate(all_text_tids, axis=0))
        result.append(np.concatenate(all_text_hidden, axis=0))
    else:
        result.append(None)
        result.append(None)
    result.append(np.array(sample_indices, dtype=int))
    return tuple(result)
