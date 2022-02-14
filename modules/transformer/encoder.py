
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.transformer.position_encoding import PositionalEncoding
from modules.transformer.word_embedding import WordEmbeddings

import math
import torch
import torch.nn as nn
from torch import Tensor
from modules.transformer.multihead_attention import MultiHeadedAttention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
           
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x, mask):
        
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o

class TransformerEncoder(nn.Module):
    def __init__(self, text_dict, max_source_positions, max_target_positions, hidden_size, ff_size, num_heads, num_layers, dropout, emb_dropout):
        super(TransformerEncoder, self).__init__()
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.padding_idx=text_dict.pad()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for num in range(num_layers)
            ]
        )
        self.word_embedding = WordEmbeddings(embedding_dim=512, vocab_size=len(text_dict), 
            pad_idx=text_dict.pad(), num_heads=8, norm_type="batch", activation_type="softsign")

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.learn_pe = nn.Embedding(self.max_source_positions + self.padding_idx + 1, 512, self.padding_idx)
        nn.init.normal_(self.learn_pe.weight, mean=0, std=0.02)
        nn.init.constant_(self.learn_pe.weight[self.padding_idx], 0)

        self.abs_pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # learn prediction
        self.embed_lengths = nn.Embedding(self.max_target_positions + 1, 512)
        nn.init.normal_(self.embed_lengths.weight, mean=0, std=0.02)


    def forward(self, word_tokens, mask):
        """
        """
        x = self.word_embedding(word_tokens, mask)        
        x = x + self.abs_pe(word_tokens) 
        # x = x + self.learn_pe(word_tokens)  # add position encoding to word embeddings
        x = self.emb_dropout(x)  # [bs, length, embed_size]
        len_tokens = self.embed_lengths(word_tokens.new(word_tokens.size(0), 1).fill_(0))
        x = torch.cat([len_tokens, x], dim=1)
        mask = torch.cat([mask.new(word_tokens.size(0), 1).fill_(1), mask], dim=1)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        x = x[:, 1:, :]
        mask = mask[:, 1:]

        predicted_lengths_logits = torch.matmul(x[:, 0, :], self.embed_lengths.weight.transpose(0, 1)).float()
        predicted_lengths_logits[:, 0] += float('-inf')   # Cannot predict the len_token
        predicted_lengths_lprobs = F.log_softmax(predicted_lengths_logits, dim=-1)
        return x, predicted_lengths_lprobs

if __name__ == "__main__":
    hidden_size = 512
    ff_size = 2048
    num_heads = 8
    dropout = 0.1
    emb_dropout = 0.1
    num_layers = 6

    m = TransformerEncoder(hidden_size, ff_size, num_heads, num_layers, dropout, emb_dropout)
    x = torch.randn(5, 100, 512)
    out = m(x, None, None)
    print(out.shape)