from torch.nn import functional as F
import torch
import time 
from model.GPT import GPT
from model.GPT_Config import GPTConfig
from Dataloader import Dataloader
from training_utils import claculate_loss,configure_optimizers
# torch.set_float32_matmul_precision('high')



#training params
max_steps=50000
lr=6e-4
Batch=8
Sequence_length=1024


device="cpu"
if torch.cuda.is_available():
    device="cuda"


with open('data/shakespeare/input.txt', 'r') as f:
    text=f.read()

dataloader=Dataloader(B=Batch,T=Sequence_length)

model = GPT(GPTConfig())
model.to(device)
model=torch.compile(model)

# optimizer =torch.optim.AdamW(model.parameters(),lr,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1)
optimizer=configure_optimizers(model,weight_decay=0.1,learning_rate=lr,betas=(0.9,0.95),device_type=device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50000,      # number of scheduler steps
    eta_min=3e-5      # minimum learning rate
)
for step in range(max_steps):
    x,y=dataloader.next_batch()
    t0=time.time()
    x,y=x.to(device),y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits=model(x)
        loss=claculate_loss(logits,y)
    loss.backward()
    #resclaes every gradient to have a maximum norm of 1.0, if the total norm of the gradients exceeds 1.0, then the gradients are rescaled to have a norm of 1.0. This is done to prevent exploding gradients and stabilize training.
    ####
    #we are basically considering the gradients as a vector and calculating the norm of that vector. If the norm exceeds 1.0, we scale down the gradients to have a norm of 1.0. This is done to prevent exploding gradients and stabilize training.
    ####
    
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    scheduler.step()
    # torch.cuda.synchronize() # wait for all kernels in all streams on a CUDA device to complete
    t1=time.time()
    dt=(t1-t0)*1000
    print(f"step{step}:|||||loss:{loss.item()},|||||norm:{norm.item():.2f},|||||tokens/sec: {dataloader.B*dataloader.T/dt*1000:.2f},||||| time: {dt:.2f}ms")
    


# print(loss)

