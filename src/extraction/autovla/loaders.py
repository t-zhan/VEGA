"""
Load AutoVLA model and dataloader from a config file.
No modification to AutoVLA source code is required.
"""

import sys
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch.utils.data import DataLoader

# Make AutoVLA importable without installing
# __file__ is src/extraction/autovla/loaders.py → .parent×4 = repo root
_AUTOVLA_ROOT = Path(__file__).parent.parent.parent.parent / "third_party" / "AutoVLA"
if str(_AUTOVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOVLA_ROOT))
if str(_AUTOVLA_ROOT / "navsim") not in sys.path:
    sys.path.insert(0, str(_AUTOVLA_ROOT / "navsim"))

from models.autovla import AutoVLA  # noqa: E402
from dataset_utils.sft_dataset import SFTDataset, DataCollator  # noqa: E402
from transformers import AutoProcessor  # noqa: E402


def load_autovla(config_path: str, checkpoint_path: str, device: str = "cuda") -> AutoVLA:
    """Instantiate AutoVLA and load weights from a Lightning checkpoint.

    Args:
        config_path: Path to training YAML config (e.g. config/training/xxx.yaml).
        checkpoint_path: Path to Lightning .ckpt file.
        device: Target device string passed to AutoVLA.

    Returns:
        AutoVLA instance in eval mode with weights loaded.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = AutoVLA(config, inference=True, device=device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    # Lightning wraps the model under "autovla." prefix in SFTAutoVLA
    state_dict = {
        k.replace("autovla.", "").replace("drivevla.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_dataloader(
    config_path: str,
    split: Literal["train", "val"] = "val",
    batch_size: int = 1,
    start: int = None,
    end: int = None,
) -> DataLoader:
    """Build a DataLoader backed by SFTDataset + DataCollator.

    Args:
        config_path: Path to training YAML config.
        split: 'train' or 'val'.
        batch_size: Number of samples per batch.
        start: Start index (inclusive). None means 0.
        end: End index (exclusive). None means len(dataset).

    Returns:
        DataLoader yielding batches compatible with Qwen2.5-VL forward signature.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"])

    dataset = SFTDataset(
        config["data"][split],
        config["model"],
        processor,
        using_cot=config["model"]["use_cot"],
    )

    if start is not None or end is not None:
        indices = range(start or 0, end if end is not None else len(dataset))
        dataset = torch.utils.data.Subset(dataset, indices)

    collator = DataCollator(
        processor=processor,
        ignore_index=config["model"]["tokens"]["ignore_index"],
        assistant_id=config["model"]["tokens"]["assistant_id"],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config["inference"]["num_workers"],
        shuffle=False,
    )
