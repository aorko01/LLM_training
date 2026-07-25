from torch.nn import functional as F
import torch
import tiktoken
from model.GPT import GPT
from model.GPT_Config import GPTConfig


device="cpu"
if torch.cuda.is_available():
    device="cuda"



def claculate_loss(logits,target)->float:
    #cross entropy expects 2D tensors so we are keeping the last dimension fixed and letting the first dimensions change accordingly 
    #(B,T,C)=>(B*T,C)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),target.view(-1))
    return loss

with open('data/shakespeare/input.txt', 'r') as f:
    text=f.read()

text= text[:1000]
enc=tiktoken.get_encoding('gpt2')
tokens=enc.encode(text)
B,T=4,32
buf = torch.tensor(tokens[:B*T+1])
x=buf[:-1].view(B,T)
y=buf[1:].view(B,T)
x = x.to(device)
y = y.to(device)

model = GPT(GPTConfig())
model.to(device)
logits=model(x)
loss=claculate_loss(logits,y)
print(loss)

