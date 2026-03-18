from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort
import torch

if TYPE_CHECKING:
    from typing import Callable


class _OnnxPolicyWithNormalizer(torch.nn.Module):
    """Export wrapper that folds obs normalization into the ONNX graph."""

    def __init__(self, actor_critic: Any, policy_normalizer: torch.nn.Module | None = None):
        super().__init__()
        self.actor_critic = actor_critic
        self.policy_normalizer = policy_normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.policy_normalizer is not None:
            obs = self.policy_normalizer(obs)
        return cast(torch.Tensor, self.actor_critic.act_inference(obs))


def _build_ort_providers() -> list[str]:
    """Prefer CUDA/CPU providers and skip TensorRT probing noise."""
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    # Fallback to whatever ORT reports if neither CUDA nor CPU is visible.
    return providers if providers else list(ort.get_available_providers())


def _to_numpy_input(obs: torch.Tensor) -> np.ndarray:
    return obs.detach().cpu().numpy().astype(np.float32, copy=False)


def _run_onnx_with_batch_support(
    session: ort.InferenceSession, input_name: str, batched_input: np.ndarray
) -> np.ndarray:
    """Run ONNX with fallback for models exported with fixed batch size = 1."""
    expected_batch = session.get_inputs()[0].shape[0]
    if isinstance(expected_batch, int) and expected_batch > 0 and batched_input.shape[0] != expected_batch:
        if expected_batch != 1:
            raise ValueError(
                f"ONNX model expects fixed batch={expected_batch}, but got batch={batched_input.shape[0]}."
            )
        outputs: list[np.ndarray] = []
        for i in range(batched_input.shape[0]):
            outputs.append(cast(np.ndarray, session.run(None, {input_name: batched_input[i : i + 1]})[0]))
        return np.concatenate(outputs, axis=0)
    return cast(np.ndarray, session.run(None, {input_name: batched_input})[0])


def export_parkour_onnx_model(*, runner, obs: torch.Tensor, export_dir: str) -> str:
    """Export a single-file ONNX policy that takes raw obs directly."""
    export_path = Path(export_dir) / "policy.onnx"
    export_model = _OnnxPolicyWithNormalizer(
        actor_critic=runner.alg.actor_critic,
        policy_normalizer=runner.normalizers.get("policy"),
    )
    export_model = export_model.to(obs.device)
    export_model.eval()
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (obs,),
            str(export_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=17,
            external_data=False,  # 单文件导出，不生成 .onnx.data
        )
    return str(export_path)


def load_parkour_onnx_model(
    model_dir: str, get_subobs_func: Callable, depth_shape: tuple, proprio_slice: slice
) -> Callable:
    """Load the ONNX model as policy, but only for parkour task setting."""
    ort_providers = _build_ort_providers()
    policy_path = os.path.join(model_dir, "policy.onnx")
    if os.path.exists(policy_path):
        policy_session = ort.InferenceSession(policy_path, providers=ort_providers)
        policy_input_name = policy_session.get_inputs()[0].name

        def policy(obs: torch.Tensor) -> torch.Tensor:
            actor_output = _run_onnx_with_batch_support(policy_session, policy_input_name, _to_numpy_input(obs))
            return torch.from_numpy(actor_output).to(obs.device)

        return policy

    encoder_path = os.path.join(model_dir, "0-depth_encoder.onnx")
    actor_path = os.path.join(model_dir, "actor.onnx")
    missing_files = [path for path in (encoder_path, actor_path) if not os.path.exists(path)]
    if missing_files:
        missing_lines = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing ONNX model files. Expected either `policy.onnx` or the legacy split export:\n"
            f"{missing_lines}"
        )

    encoder = ort.InferenceSession(encoder_path, providers=ort_providers)
    actor = ort.InferenceSession(actor_path, providers=ort_providers)
    encoder_input_name = encoder.get_inputs()[0].name
    actor_input_name = actor.get_inputs()[0].name

    def policy(obs: torch.Tensor) -> torch.Tensor:
        depth_image_input = get_subobs_func(obs)
        depth_image_input = _to_numpy_input(depth_image_input)
        depth_image_input = depth_image_input.reshape((-1, *depth_shape))
        depth_image_output = _run_onnx_with_batch_support(encoder, encoder_input_name, depth_image_input)
        actor_input = np.concatenate(
            [
                _to_numpy_input(obs)[:, proprio_slice],
                depth_image_output,
            ],
            axis=1,
        )
        actor_output = _run_onnx_with_batch_support(actor, actor_input_name, actor_input)
        return torch.from_numpy(actor_output).to(obs.device)

    return policy
