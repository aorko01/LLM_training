import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,config):
        super.__init__()
        self.ln1=nn.Linear(config.n_embd,config.n_embd*4)
        self.glu=nn.GELU()
        self.ln2=nn.Linear(4*config.n_embd,config.n_embd)
        self.drop=nn.Dropout(config.drop_prob)
        
    def forward(x);
        x=self.ln1(x)
        x=self.glu(x)
        x=self.ln2(x)
        x=self.drop(x)
        
        return x