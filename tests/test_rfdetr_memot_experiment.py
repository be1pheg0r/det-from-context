"""Static checks for the runnable RF-DETR + MeMOT experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.rfdetr_memot.submit_datasphere import _load_paths
from omegaconf import OmegaConf

from context_detection.config import load_config


def test_experiment_config_matches_video_classes_and_external_memot() -> None:
    config = load_config("experiments/rfdetr_memot/config.yaml")
    raw: Any = OmegaConf.to_container(
        OmegaConf.load("datasets/video_dataloader/config.yaml"), resolve=True
    )
    assert isinstance(raw, dict) and isinstance(raw.get("classes"), dict)

    assert config.context.name == "memot"
    assert config.data.name == "video_dataloader"
    assert config.data.clip_len >= 2
    assert config.detector.num_classes == len(raw["classes"])
    assert not config.train.denoising
    assert config.train.amp_dtype == "fp16"
    assert not config.train.use_ema
    assert config.data.splits == ["train", "validation"]
    assert config.logging.max_visual_images == 6
    assert config.logging.max_diagnostic_images == 256


def test_datasphere_paths_and_template_use_independent_video_roots() -> None:
    root = Path("experiments/rfdetr_memot")
    paths = _load_paths(root / "paths.yaml")
    template = (root / "datasphere_job.template.yaml").read_text(encoding="utf-8")

    assert paths["videos_dir"] != paths["annotations_dir"]
    assert "__VIDEO_DATASET_VIDEOS_DIR__" in template
    assert "__VIDEO_DATASET_ANNOTATIONS_DIR__" in template
    assert "agent_specs" not in template
