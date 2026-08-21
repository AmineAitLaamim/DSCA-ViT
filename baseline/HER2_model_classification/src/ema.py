from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.copy_(ema_value * self.decay + model_value * (1.0 - self.decay))
            else:
                ema_value.copy_(model_value)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state):
        self.ema.load_state_dict(state)

