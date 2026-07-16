import math 
import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.Wq=nn.Linear(config.n_embd,config.n_embd)
        self.Wk=nn.Linear(config.n_embd,config.n_embd)
        self.Wv=nn.Linear(config.n_embd,config.n_embd)
        
        self.output=nn.Linear(config.n_embd,config.n_embd)
        self.attn_dropout=nn.Dropout(config.drop_prob)
        self.output_dropout=nn.Dropout(config.drop_prob)
        self.dropout=config.drop_prob
        
        self.n_head=config.n_head
        
    def forward(self,X):
        Batch,Token,Embedding=X.shape
        
        Q=self.Wq(X)
        K=self.Wk(X)
        V=self.Wv(X)
        head_dim=Embedding//self.n_head
        
        Q=Q.view(Batch,Token,self.n_head,head_dim).transpose(-2,-3)
        K=K.view(Batch,Token,self.n_head,head_dim).transpose(-2,-3)
        V=V.view(Batch,Token,self.n_head,head_dim).transpose(-2,-3)
        
        attention= Q@ K.transpose(-2,-1) * (1.0/math.sqrt(K.size(-1)))
        attention=F.softmax(attention,dim=-1)
        # apply droput 
        attention=self.attn_dropout(attention)
        
        y=attention @ V 
        # we are doing the contiguous because the transpose would break the contiguity 
        y=y.transpose(-2,-3).contiguous()
        y=y.view(Batch,Token,Embedding)
        # or we can directly do reshpae that is essentially perform similar in this case but contiguous-> view still better 
        # y=output.reshpae(Batch,Token,Embedding)
        y=output(y)
        y=output_dropout(output)
               
        return output
        
        
        
        
