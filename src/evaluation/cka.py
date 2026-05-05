from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """선형 커널 (Gram matrix).

    Args:
        x: 특성 행렬, shape (n, d).

    Returns:
        Gram matrix, shape (n, n).
    """
    return x @ x.T


def rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """RBF 커널.

    Args:
        x: 특성 행렬, shape (n, d).
        sigma: 커널 대역폭. None이면 중간값 휴리스틱.

    Returns:
        Gram matrix, shape (n, n).
    """
    dist_sq = torch.cdist(x, x, p=2) ** 2
    if sigma is None:
        sigma_tensor = torch.sqrt(torch.median(dist_sq[dist_sq > 0]) / 2)
    else:
        sigma_tensor = torch.tensor(sigma, device=x.device, dtype=x.dtype)
    return torch.exp(-dist_sq / (2 * sigma_tensor**2 + 1e-8))


def hsic(k_x: torch.Tensor, k_y: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion (편향 추정).

    Args:
        k_x: 첫 번째 Gram matrix, shape (n, n).
        k_y: 두 번째 Gram matrix, shape (n, n).

    Returns:
        HSIC 값 (스칼라).
    """
    n = k_x.shape[0]
    h = torch.eye(n, device=k_x.device) - 1.0 / n
    return torch.trace(k_x @ h @ k_y @ h) / ((n - 1) ** 2)


def cka(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = "linear",
) -> float:
    """Centered Kernel Alignment (CKA).

    두 표현 공간의 유사도를 측정. 값 범위: [0, 1].
    1에 가까울수록 유사한 표현.

    Args:
        x: 첫 번째 표현, shape (n, d1).
        y: 두 번째 표현, shape (n, d2).
        kernel: "linear" 또는 "rbf".

    Returns:
        CKA 값 [0, 1].

    Raises:
        ValueError: x, y의 샘플 수가 다를 때.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"샘플 수 불일치: x={x.shape[0]}, y={y.shape[0]}")

    kernel_fn = linear_kernel if kernel == "linear" else rbf_kernel

    k_x = kernel_fn(x)
    k_y = kernel_fn(y)

    hsic_xy = hsic(k_x, k_y)
    hsic_xx = hsic(k_x, k_x)
    hsic_yy = hsic(k_y, k_y)

    denom = torch.sqrt(hsic_xx * hsic_yy + 1e-10)
    return float((hsic_xy / denom).item())


class CKAAnalyzer:
    """모델 적응 전후 표현 변화 분석.

    각 레이어의 CKA를 계산하여 어디서 표현이 가장 많이 변화했는지 측정.

    Args:
        model: PyTorch 모델.
        layer_patterns: 분석할 레이어 이름 패턴 리스트.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_patterns: list[str] | None = None,
    ) -> None:
        self.model: nn.Module = model
        self.layer_patterns: list[str] = layer_patterns or []
        self._hooks: list[RemovableHandle] = []
        self._activations: dict[str, torch.Tensor] = {}

    def _get_hook(self, name: str):
        def hook_fn(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            _ = module
            _ = inputs
            if isinstance(output, tuple):
                output = output[0]
            # HuggingFace ModelOutput 객체(BaseModelOutputWithPastAndCrossAttentions 등)
            # 는 last_hidden_state 속성이 텐서를 담고 있음.
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                output = output.hidden_states[-1]
            if not isinstance(output, torch.Tensor):
                # detach 가능한 텐서가 아니면 스킵 (dict, dataclass 등)
                return
            self._activations[name] = output.detach().cpu()

        return hook_fn

    def register_hooks(self) -> None:
        """분석 대상 레이어에 forward hook 등록."""
        self.remove_hooks()
        for name, module in self._iter_named_modules():
            if any(pattern in name for pattern in self.layer_patterns):
                hook = module.register_forward_hook(self._get_hook(name))
                self._hooks.append(hook)
                logger.debug(f"Hook 등록: {name}")

    def _iter_named_modules(self) -> list[tuple[str, nn.Module]]:
        modules: list[tuple[str, nn.Module]] = []
        stack: list[tuple[str, nn.Module]] = [("", self.model)]

        while stack:
            prefix, module = stack.pop()
            for child_name, child_module in module.named_children():
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                modules.append((full_name, child_module))
                stack.append((full_name, child_module))

        return modules

    def remove_hooks(self) -> None:
        """등록된 hook 모두 제거."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_activations(self) -> dict[str, torch.Tensor]:
        """현재 저장된 활성화 반환."""
        return dict(self._activations)

    def clear_activations(self) -> None:
        """저장된 활성화 초기화."""
        self._activations.clear()

    def compare_representations(
        self,
        activations_before: dict[str, torch.Tensor],
        activations_after: dict[str, torch.Tensor],
        kernel: str = "linear",
    ) -> dict[str, float]:
        """적응 전후 표현 유사도를 레이어별로 계산.

        Args:
            activations_before: 적응 전 활성화.
            activations_after: 적응 후 활성화.
            kernel: CKA 커널 유형.

        Returns:
            레이어 이름 → CKA 값 딕셔너리.
        """
        results: dict[str, float] = {}

        common_layers = set(activations_before.keys()) & set(activations_after.keys())

        for layer_name in sorted(common_layers):
            before = activations_before[layer_name]
            after = activations_after[layer_name]

            # Flatten to 2D: (batch*seq, hidden)
            before_2d = before.reshape(-1, before.shape[-1])
            after_2d = after.reshape(-1, after.shape[-1])

            # 샘플 수 맞추기
            min_n = min(before_2d.shape[0], after_2d.shape[0], 1000)
            before_2d = before_2d[:min_n]
            after_2d = after_2d[:min_n]

            cka_val = cka(before_2d, after_2d, kernel=kernel)
            results[layer_name] = cka_val
            logger.debug(f"CKA({layer_name}): {cka_val:.4f}")

        return results
