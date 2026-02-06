"""
Emotion-conditioned model wrapper with TraitBasis-style activation injection.

Projects 4D emotion vectors to hidden_dim and adds them to hidden states at a
chosen layer during the forward pass. Compatible with Unsloth/Llama + LoRA.
"""

import torch
import torch.nn as nn
from typing import Optional, Any


def _find_layers_module(model) -> nn.ModuleList:
    """Find the transformer layers in a model (handles PEFT, Unsloth, etc.)."""
    for attr in ("model", "base_model"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        for sub in ("model", "model"):
            subm = getattr(m, sub, None)
            if subm is not None and hasattr(subm, "layers"):
                return subm.layers
        if hasattr(m, "layers"):
            return m.layers
    raise ValueError("Could not find model.layers - unsupported model structure")


class EmotionInjector(nn.Module):
    """
    Injects projected emotion vectors into hidden states at a specific layer.
    Uses a forward hook for minimal model modification.
    """

    def __init__(
        self,
        emotion_dim: int = 4,
        hidden_size: int = 4096,
        layer_idx: int = 16,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.emotion_dim = emotion_dim
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.proj = nn.Linear(emotion_dim, hidden_size)
        self._current_emotion: Optional[torch.Tensor] = None
        self._hook_handle = None

    def set_emotion(self, emotion: Optional[torch.Tensor]):
        """Set emotion for the next forward pass. Shape: (batch, 4) or (4,)."""
        self._current_emotion = emotion

    def _hook_fn(self, module: nn.Module, input: tuple, output: Any) -> Any:
        if self._current_emotion is None:
            return output
        emotion = self._current_emotion
        if emotion.dim() == 1:
            emotion = emotion.unsqueeze(0)
        device = output.device
        if emotion.device != device:
            emotion = emotion.to(device)
        projected = self.proj(emotion)
        return output + self.alpha * projected.unsqueeze(1)

    def register_hook(self, model: nn.Module) -> None:
        """Register forward hook on the target layer."""
        layers = _find_layers_module(model)
        if self.layer_idx >= len(layers):
            raise ValueError(
                f"layer_idx={self.layer_idx} >= num_layers={len(layers)}"
            )
        target = layers[self.layer_idx]
        self._hook_handle = target.register_forward_hook(self._hook_fn)

    def remove_hook(self) -> None:
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class EmotionConditionedWrapper(nn.Module):
    """
    Wraps a causal LM to accept an optional 'emotion' kwarg and inject it
    at a chosen layer via EmotionInjector.
    """

    def __init__(self, model: nn.Module, injector: EmotionInjector):
        super().__init__()
        self.model = model
        self.injector = injector
        self.injector.register_hook(model)

    @property
    def peft_config(self):
        """Delegate to inner model so Unsloth/TRL trainer recognizes this as a PEFT model."""
        return getattr(self.model, "peft_config", None)

    @property
    def base_model(self):
        """Delegate to inner model for PEFT compatibility."""
        return getattr(self.model, "base_model", self.model)

    def forward(self, input_ids=None, **kwargs):
        emotion = kwargs.pop("emotion", None)
        self.injector.set_emotion(emotion)
        return self.model(input_ids=input_ids, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _get_hidden_size(model: nn.Module) -> int:
    """Extract hidden_size from model config (handles PEFT, Unsloth, etc.)."""
    config = getattr(model, "config", None)
    if config is None:
        config = getattr(model.model, "config", None)
    if config is None and hasattr(model, "base_model"):
        config = getattr(model.base_model, "config", None)
    if config is None and hasattr(model, "model"):
        config = getattr(model.model.model, "config", None)
    return getattr(config, "hidden_size", 4096) if config else 4096


def attach_emotion_injector(
    model: nn.Module,
    emotion_dim: int = 4,
    layer_idx: int = 16,
    alpha: float = 0.1,
) -> nn.Module:
    """
    Attach emotion injector to a PEFT model WITHOUT wrapping it.
    Use this for training so Unsloth/TRL recognize the model as PEFT.

    The model gets an `injector` attribute. EmotionDPOTrainer sets
    model.injector.set_emotion(emotion) before each forward pass.

    Args:
        model: PEFT model (e.g. Unsloth LoRA model)
        emotion_dim: Dimension of emotion vector (default 4 for I,T,A,S)
        layer_idx: Layer index for injection (default 16 = middle of 32 layers)
        alpha: Scaling factor for injection (default 0.1)

    Returns:
        The same model with injector attached (in-place modification).
    """
    hidden_size = _get_hidden_size(model)
    injector = EmotionInjector(
        emotion_dim=emotion_dim,
        hidden_size=hidden_size,
        layer_idx=layer_idx,
        alpha=alpha,
    )
    injector = injector.to(next(model.parameters()).device)
    injector.register_hook(model)
    model.injector = injector
    return model


def wrap_model_with_emotion(
    model: nn.Module,
    emotion_dim: int = 4,
    layer_idx: int = 16,
    alpha: float = 0.1,
) -> EmotionConditionedWrapper:
    """
    Wrap a model with emotion conditioning (for inference).
    For DPO training, use attach_emotion_injector() instead so the trainer
    recognizes the model as PEFT.

    Args:
        model: Causal LM (e.g. Unsloth FastLanguageModel or LlamaForCausalLM)
        emotion_dim: Dimension of emotion vector (default 4 for I,T,A,S)
        layer_idx: Layer index for injection (default 16 = middle of 32 layers)
        alpha: Scaling factor for injection (default 0.1)

    Returns:
        EmotionConditionedWrapper that accepts emotion= in forward()
    """
    hidden_size = _get_hidden_size(model)
    injector = EmotionInjector(
        emotion_dim=emotion_dim,
        hidden_size=hidden_size,
        layer_idx=layer_idx,
        alpha=alpha,
    )
    injector = injector.to(next(model.parameters()).device)
    return EmotionConditionedWrapper(model, injector)
