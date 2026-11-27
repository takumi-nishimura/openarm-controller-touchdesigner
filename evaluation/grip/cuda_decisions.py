"""Native CUDA inference for the Grip benchmark."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from evaluation.grip.input_images import ImageInput

CUDA_CHECKPOINTS = {
    "d1-2b": ("LiquidAI/d1-2B", "a6bce774e6dde338dcc3036ceb5d91f78bf4c19e"),
    "intern-decision-2b": (
        "internlm/Intern-Decision-2B",
        "85c5657708a09884ff9fd8d7f57789b410a97b33",
    ),
}
CHECKPOINT_FILES = [
    "*.py",
    "*.json",
    "*.jinja",
    "*.safetensors",
    "merges.txt",
    "LICENSE*",
]

NativePredictor = Callable[[dict[str, Any], Sequence[ImageInput]], dict[str, Any]]


@dataclass(frozen=True)
class CudaDecisionModel:
    model: str
    settings: dict[str, str]
    predict: NativePredictor


def _load_liquid(checkpoint: Path, device: str) -> NativePredictor:
    import torch
    from PIL import Image
    from transformers import AutoModel

    model = (
        AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    def predict(
        questions: dict[str, Any], images: Sequence[ImageInput]
    ) -> dict[str, Any]:
        media = []
        for image in images:
            with Image.open(BytesIO(image.rgb_bytes)) as source:
                media.append(source.convert("RGB"))
        return model.system_one(None, questions, media=media)

    return predict


def _load_intern(checkpoint: Path, device: str) -> NativePredictor:
    # Use the checkpoint's native decision compiler and temperature calibration.
    engine_class = runpy.run_path(str(checkpoint / "inference.py"))["DecisionEngine"]
    engine = engine_class(checkpoint=checkpoint, device=device, dtype="bfloat16")

    def predict(
        questions: dict[str, Any], images: Sequence[ImageInput]
    ) -> dict[str, Any]:
        # The native engine accepts local paths. Keep source paths and IDs private.
        with TemporaryDirectory(prefix="grip-images-") as scratch:
            paths = []
            for index, image in enumerate(images):
                path = Path(scratch) / f"{index}.img"
                path.write_bytes(image.rgb_bytes)
                paths.append(str(path))
            return engine.predict(
                {"state": None, "questions": questions, "images": paths}
            )

    return predict


def load_cuda_model(name: str) -> CudaDecisionModel:
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "CUDA backends need Python 3.12. Use uv run --script run-grip-cuda.py "
            "from the repository root."
        )

    import torch
    import transformers
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "A BF16-capable CUDA device is required for Grip decision models."
        )

    device_index = torch.cuda.current_device()
    device = f"cuda:{device_index}"
    model, revision = CUDA_CHECKPOINTS[name]
    checkpoint = Path(
        snapshot_download(model, revision=revision, allow_patterns=CHECKPOINT_FILES)
    )
    load = _load_liquid if name == "d1-2b" else _load_intern
    native_predict = load(checkpoint, device)

    @torch.inference_mode()
    def predict(
        questions: dict[str, Any], images: Sequence[ImageInput]
    ) -> dict[str, Any]:
        result = native_predict(questions, images)
        torch.cuda.synchronize(device_index)
        return result

    settings = {
        "revision": revision,
        "device": device,
        "gpu": torch.cuda.get_device_name(device_index),
        "dtype": "bfloat16",
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
    }
    try:
        settings["flash_linear_attention"] = version("flash-linear-attention")
    except PackageNotFoundError:
        pass
    return CudaDecisionModel(model, settings, predict)
