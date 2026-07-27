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
Batch=4
Sequence_length=1024
Total_batches=524288


assert Total_batches%(Batch*Sequence_length)==0, "Total_batches must be divisible by Batch*Sequence_length"
grad_accumulation_steps=Total_batches//(Batch*Sequence_length)
print(f"grad_accumulation_steps: {grad_accumulation_steps}")

device="cpu"
if torch.cuda.is_available():
    device="cuda"

#
#Load the data
#
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

#
#Training loop
#
for step in range(max_steps):
    t0=time.time()
    optimizer.zero_grad(set_to_none=True) 
    loss_accum= 0.0
    #optimization over gradient accumulation steps. We are accumulating gradients over multiple batches before updating the model parameters. This is done to simulate a larger batch size and stabilize training.
    for micro_step in range(grad_accumulation_steps):
        x,y=dataloader.next_batch()
        x,y=x.to(device),y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits=model(x)
            loss=claculate_loss(logits,y)
            loss_accum+=loss.detach()
            #this is important for the gradient accumulation. We are dividing the loss by the number of gradient accumulation steps to average the gradients over multiple batches. This is done to simulate a larger batch size and stabilize training.
            loss=loss/grad_accumulation_steps
        loss.backward()
    #resclaes every gradient to have a maximum norm of 1.0, if the total norm of the gradients exceeds 1.0, then the gradients are rescaled to have a norm of 1.0. This is done to prevent exploding gradients and stabilize training.
    ####
    #we are basically considering the gradients as a vector and calculating the norm of that vector. If the norm exceeds 1.0, we scale down the gradients to have a norm of 1.0. This is done to prevent exploding gradients and stabilize training.
    ####
    
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize() # wait for all kernels in all streams on a CUDA device to complete
    t1=time.time()
    dt=(t1-t0)*1000
    print(f"step{step}:|||||loss:{loss_accum.item()/grad_accumulation_steps}|||||norm:{norm.item()/grad_accumulation_steps:.2f}|||||tokens/sec: {dataloader.B*dataloader.T*1000*grad_accumulation_steps/dt:.2f}||||| time: {dt:.2f}ms")
    


# print(loss)

