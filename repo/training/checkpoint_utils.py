"""Checkpoint loading and compatibility checks shared by training and evaluation."""

from pathlib import Path

import torch


def load_checkpoint(path, map_location="cpu"):
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=map_location, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dictionary: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint is missing model_state_dict: {checkpoint_path}")
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError(f"checkpoint model_state_dict must be a dictionary: {checkpoint_path}")
    return checkpoint


def validate_num_edges(checkpoint, expected_num_edges):
    checkpoint_num_edges = checkpoint.get("num_edges")
    if checkpoint_num_edges is None:
        return
    if int(checkpoint_num_edges) != int(expected_num_edges):
        raise ValueError(
            "checkpoint/model mismatch: checkpoint num_edges="
            f"{checkpoint_num_edges}, requested num_neighbors={expected_num_edges}"
        )


def load_model_weights(model, checkpoint, checkpoint_path):
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"checkpoint model_state_dict is incompatible with the requested model: {checkpoint_path}"
        ) from error


def require_resume_state(checkpoint, checkpoint_path):
    required = ("step", "epoch", "optimizer_state_dict")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f"resume checkpoint is missing {', '.join(missing)}: {checkpoint_path}; "
            "use --init_checkpoint for a weights-only checkpoint"
        )


def checkpoint_metadata(checkpoint):
    optimizer_schedule = checkpoint.get("optimizer_schedule")
    if isinstance(optimizer_schedule, dict):
        optimizer_schedule = {
            "name": _json_scalar(optimizer_schedule.get("name")),
            "factor": _json_scalar(optimizer_schedule.get("factor")),
            "warmup_steps": _json_scalar(optimizer_schedule.get("warmup_steps")),
        }
    else:
        optimizer_schedule = None
    return {
        "num_edges": _json_scalar(checkpoint.get("num_edges")),
        "noise_level": _json_scalar(checkpoint.get("noise_level")),
        "step": _json_scalar(checkpoint.get("step")),
        "epoch": _json_scalar(checkpoint.get("epoch")),
        "has_optimizer_state": "optimizer_state_dict" in checkpoint,
        "optimizer_schedule": optimizer_schedule,
    }


def _json_scalar(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)
