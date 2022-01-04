"""
taken from: https://github.com/karpathy/minBERT/
BERT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import top_k_top_p_filtering
from modules.transformer.axial_attention import AxialBlock


logger = logging.getLogger(__name__)


class BERTConfig:
    """ base BERT config, params common to all BERT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, shape, **kwargs):
        self.vocab_size = vocab_size
        self.shape = shape
        for k,v in kwargs.items():
            setattr(self, k, v)



class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = AxialBlock(config.n_embd, config.n_head)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        attn = self.attn(self.ln1(x))
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class CBERT(nn.Module):
    """  the full BERT language model, with a context size of block_size """
    def __init__(self, vocab_size, shape, n_class, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, label_cond=True,
                 hw_separation=True, temporal_separation=True, use_cls_token=True, initializer_range=0.02,
                 
                 ):
        super().__init__()
        self.length, self.height, self.width = shape
        config = BERTConfig(vocab_size=vocab_size, shape=shape,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, label_cond=label_cond)
        # input embedding stem
        self.label_cond = label_cond
        # self.cls_emb = nn.Embedding(n_class, config.n_embd)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        self.hw_separation = hw_separation
        if hw_separation:
            assert temporal_separation
            if use_cls_token:
                self.cls_pos_embedding = nn.Parameter(initializer_range * torch.randn(1, config.n_embd))
            initializer_range = initializer_range / math.sqrt(3)
            self.temporal_pos_embedding = nn.Parameter(initializer_range * torch.randn(self.length, config.n_embd))
            self.height_pos_embedding = nn.Parameter(initializer_range * torch.randn(self.height, config.n_embd))
            self.width_pos_embedding = nn.Parameter(initializer_range * torch.randn(self.width, config.n_embd))
        else:
            block_size = self.length * self.height* self.width
            self.pos_emb = nn.Parameter(torch.zeros(1, block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, token_embeddings, extract_feature=False):
        """[bs, 4, 32, 32, 256]
        """
        # forward the BERT model
        # token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if self.hw_separation:
            position_embeddings = (self.temporal_pos_embedding.reshape(self.length, 1, 1, -1) +
                           self.height_pos_embedding.reshape(1, self.height, 1, -1) +
                           self.width_pos_embedding.reshape(1, 1, self.width, -1))
            position_embeddings = position_embeddings.unsqueeze(0)
        else:
            t = token_embeddings.shape[1]
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)
        x = self.ln_f(x)
        if extract_feature:
            return x

        logits = self.head(x)
        return logits



def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


if __name__ == "__main__":
    model = CBERT(vocab_size=512, shape=(4, 32, 32), n_class=1000, n_layer=12, n_head=8, n_embd=256)
    x = torch.randint(0, 512, (2, 4, 32, 32), dtype=torch.long)

    logits = model(x)

    print(logits.shape)



