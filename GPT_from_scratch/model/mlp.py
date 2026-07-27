import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.gelu=nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd, bias=config.bias)
        self.drop=nn.Dropout(config.dropout)
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        x=self.drop(x)
        return x