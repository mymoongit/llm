'''
Transformer


'''

import torch
import torch.nn as nn

from posision import PositionalEncoding
from encoder import EncoderLayer
from decoder import DecoderLayer


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(-1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        encoder_embedding = self.encoder_embedding(src)
        en_positional_embedding = self.positional_encoding(encoder_embedding)
        src_embedded = self.dropout(en_positional_embedding)

        decoder_embedding = self.decoder_embedding(tgt)
        de_positional_embedding = self.positional_encoding(decoder_embedding)
        tgt_embedded = self.dropout(de_positional_embedding)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)


        output = self.linear(dec_output)
        return output


src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = d_model * 4
max_len = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)


src_data = torch.randint(1, src_vocab_size, (5, max_len))  # (batch_size, seq_length)
tgt_data = torch.randint(1, src_vocab_size, (5, max_len))

output = transformer(src_data, tgt_data[:, :10])

print(output.shape)









