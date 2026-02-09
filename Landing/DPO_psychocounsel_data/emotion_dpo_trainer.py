"""
Custom DPOTrainer that passes emotion to the model during forward passes.

Overrides concatenated_forward to set emotion on model.injector before the
forward pass. The model must have an `injector` attribute (from
attach_emotion_injector). Emotion is duplicated for the concatenated batch
(chosen + rejected) since both use the same per-prompt emotion.
"""

import torch
from trl import DPOTrainer


class EmotionDPOTrainer(DPOTrainer):
    """
    DPOTrainer that passes `emotion` from the batch to the model's injector.
    Requires batch to contain "emotion" key of shape (batch_size, 4).
    Model must have model.injector (from attach_emotion_injector).
    """

    def concatenated_forward(
        self, model, batch, is_ref_model: bool = False
    ):
        """Set emotion on model.injector before the forward pass.

        The policy model has an injector (attached via attach_emotion_injector).
        We set emotion to the concatenated tensor [chosen_emotions, rejected_emotions]
        so the single forward over the concatenated batch uses the correct
        emotion per example. Ref model does not get emotion.
        """
        emotion = batch.get("emotion") if not is_ref_model else None
        if emotion is not None and hasattr(model, "injector"):
            # Duplicate for concatenated batch: [chosen_0...chosen_N, rejected_0...rejected_N]
            emotion_cat = torch.cat([emotion, emotion], dim=0)
            device = next(model.parameters()).device
            model.injector.set_emotion(emotion_cat.to(device))

        return super().concatenated_forward(model, batch, is_ref_model)

    def create_optimizer(self):
        """Override to ensure injector parameters are trainable."""

        # Get default optimizer first
        optimizer = super().create_optimizer()

        # Verify injector params are included
        if hasattr(self.model, 'injector'):
            injector_params = set(id(p) for p in self.model.injector.parameters())
            optimizer_params = set(id(p) for group in optimizer.param_groups for p in group['params'])

            missing = injector_params - optimizer_params
            if missing:
                print(f"⚠️  WARNING: {len(missing)} injector params NOT in optimizer!")
                print("   Adding them manually...")

                # Add missing params to optimizer
                optimizer.add_param_group({
                    'params': [p for p in self.model.injector.parameters() if id(p) in missing],
                    'lr': self.args.learning_rate,
                })
                print(f"   ✅ Added {len(missing)} injector params to optimizer")
            else:
                print(f"   ✅ All {len(injector_params)} injector params in optimizer")

        return optimizer
