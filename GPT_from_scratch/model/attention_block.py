import torch 
import torch.nn as nn
import Attention from attention
import LayerNorm from layer_norm
import MLP from mlp


class Attention_Block(nn.Module):
    def __init__(self,config):
        self.lm1=LayerNorm(config.n_embd,bias=config.bias)
        self.attn=Attention(config)
        self.lm2=LayerNorm(config.n_embd,bias=config.bias)
        self.mlp=MLP(config)
        
    def forward(self,x):
        x =lm1(x)
        x=attn(x)
        x=lm2(x)
        x=mlp(x)
        
        return x
        