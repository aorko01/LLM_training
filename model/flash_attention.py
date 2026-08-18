import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..model.attention_registry import BaseAttention, register_attention

@register_attention("flash")
class Attention(BaseAttention):
    def __init__(self, config):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, X):
        Batch, Token, Embedding = X.shape
        
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        
        # Reshape for multi-head attention
        Q = Q.view(Batch, Token, self.n_head, self.head_dim)
        K = K.view(Batch, Token, self.n_head, self.head_dim)
        V = V.view(Batch, Token, self.n_head, self.head_dim)
        
        # Use Flash Attention when available
        try:
            # PyTorch 2.0+ scaled_dot_product_attention with Flash Attention backend
            # Supports causal masking and dropout natively
            y = F.scaled_dot_product_attention(
                Q.transpose(1, 2),  # [Batch, n_head, Token, head_dim]
                K.transpose(1, 2),
                V.transpose(1, 2),
                attn_mask=None,  # is_causal handles the masking
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        except RuntimeError:
            # Fallback to manual implementation if Flash Attention fails
            Q = Q.transpose(1, 2)  # [Batch, n_head, Token, head_dim]
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            scores = scores.masked_fill(
                ~self.causal_mask[:, :, :Token, :Token],
                float("-inf"),
            )
            attention = F.softmax(scores, dim=-1)
            attention = self.attn_dropout(attention)
            y = attention @ V
            
        # Reshape back
        y = y.transpose(1, 2).contiguous()
        y = y.view(Batch, Token, Embedding)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y