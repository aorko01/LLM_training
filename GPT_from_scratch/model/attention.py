import math 
import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.Wq=nn.Linear(config.n_embd,config.n_embd, bias=config.bias)
        self.Wk=nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.Wv=nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_proj=nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout=nn.Dropout(config.dropout)
        self.resid_dropout=nn.Dropout(config.dropout)
        self.n_head=config.n_head
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self,X):
        Batch,Token,Embedding=X.shape
        Q=self.Wq(X)
        K=self.Wk(X)
        V=self.Wv(X)
        head_dim=Embedding//self.n_head
        Q=Q.view(Batch,Token,self.n_head,head_dim).transpose(-2,-3)
        K=K.view(Batch,Token,self.n_head,head_dim).transpose(-2,-3)
        V=V.view(Batch,Token,self.n_head,head_dim).transpose(-2,-3)
        scores=(Q@ K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :Token, :Token],
            float("-inf"),
        )
        attention=F.softmax(scores,dim=-1)
        attention=self.attn_dropout(attention)

        y=attention @ V
        y=y.transpose(-2, -3).contiguous()
        y=y.view(Batch, Token, Embedding)
        y=self.c_proj(y)
        y=self.resid_dropout(y)
        return y