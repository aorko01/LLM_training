import torch 
import torch.nn as nn
from .attention import Attention 
from .layer_norm import LayerNorm 
from .mlp import MLP 


class Attention_Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lm1=LayerNorm(config.n_embd,bias=config.bias)
        self.attn=Attention(config)
        self.lm2=LayerNorm(config.n_embd,bias=config.bias)
        self.mlp=MLP(config)
        
    def forward(self,x):
        # x =self.lm1(x)
        # x=self.attn(x)
        # x=self.lm2(x)
        # x=self.mlp(x)
        #missed the residual stream and the loss was dramartically higher. can uncomment to check how the loss gets saturated 
        x = x + self.attn(self.lm1(x))
        x = x + self.mlp(self.lm2(x))
        return x
        