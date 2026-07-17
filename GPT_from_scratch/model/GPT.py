import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self,config):
        super.__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config=config
        self.transformer = nn.ModuleDict(dict(
            #vocab-> embedding
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.vocab_size,config.n_embd),
            drop=nn.Dropout(config.dropout),
            attn_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_=LayerNorm(config.n_embd,bias=config.bias)
        ))
        #embedding->vocab
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #wte matrix projects the vocab to the embedding and at the output layer we project the embedding back to vocab. So , we share the parameter in both layers as they are essentially doing the same thing
        self.transformer.wte.weight=self.lm_head.weight
        
        self.apply(self.__init__weights)
        # apply() is defined inside nn.Module, so your model inherits it.
        # Conceptually, nn.Module looks something like this (greatly simplified):
        # class Module:

        #     def apply(self, fn):
        #         for child in self.children():
        #             child.apply(fn)

        #         fn(self)
        #         return self
        # apply special scaled init to the residual projections, per GPT-2 paper
        #This is done for the stability issue that occurs due to residual connections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        logits=self.lm_head(x)
        
        return logits
        