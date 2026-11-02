from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RealManSmolVLATrainConfig:
    dataset: Path = REPO_ROOT / "datasets" / "realman_tape_red_block_auto_v2"
    repo_id: str | None = None
    output_root: Path = REPO_ROOT / "outputs" / "train"
    job_name: str = "realman_tape_red_block_auto_v2_smolvla_7d"

    # Start conservatively. The RealMan Quest dataset is 20 Hz by default, and
    # 50 action steps cover about 2.5 s.
    steps: int = 20000
    batch_size: int = 16
    num_workers: int = 8
    log_freq: int = 1000
    save_freq: int = 1000
    seed: int = 1000

    device: str = "cuda"
    amp: bool = True
    use_imagenet_stats: bool = True
    tolerance_s: float = 1e-4

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32
    image_size: int = 256
    num_vlm_layers: int = 16
    num_expert_layers: int = -1
    expert_width_multiplier: float = 0.75

    hf_home: Path = REPO_ROOT / ".cache" / "huggingface"


CFG = RealManSmolVLATrainConfig()
