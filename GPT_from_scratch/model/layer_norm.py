import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()

        self.ln = nn.LayerNorm(ndim, eps=eps)

        if not bias:
            self.ln.register_parameter("bias", None)

    def forward(self, x):
        return self.ln(x)