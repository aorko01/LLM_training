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
        
        self.n_head=config.n_head
        
    def forward(self,X):
        Batch,Token,Context=X.shape
        
        Q=self.Wq(X)
        K=self.Wk(X)
        V=self.Wv(X)
        
        Q=Q.view(Batch,Token,self.n_head,Context//self.n_head).transpose(-2,-3)
        K=K.view(Batch,Token,self.n_head,Context//self.n_head).transpose(-2,-3)
        V=V.view(Batch,Token,self.n_head,Context//self.n_head).transpose(-2,-3)
        
        attention= Q@ K.transpose(-2,-1) * (1.0/math.sqrt(K.size(-1)))
        attention=F.softmax(attention,dim=-1)
        output=attention @ V 
               
        return output
        
        
        
        
