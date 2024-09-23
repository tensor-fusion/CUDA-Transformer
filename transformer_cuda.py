import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import math

attention_cuda = load(
    name="attention_cuda",
    sources=["multi_head_attention.cu"],
    verbose=True,
)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        attn_output = attention_cuda.multi_head_attention(q, k, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)

        return output


class FFN(nn.Module):
    def __init__(self, d_model, dim_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FFN(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        attn_output = self.self_attn(src)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FFN(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        attn_output = self.self_attn(tgt)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        attn_output = self.cross_attn(tgt)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)

        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_ff=2048,
        dropout=0.1,
        vocab_size=10000,
        max_seq_len=512,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.pos_decoder = PositionalEncoding(d_model, max_seq_len)

        self.encoder_layers = nn.ModuleList(
            [
                Encoder(d_model, num_heads, dim_ff, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                Decoder(d_model, num_heads, dim_ff, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory)

        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory)

        logits = self.fc_out(output)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    src_seq_len = 10
    tgt_seq_len = 10
    vocab_size = 1000

    # Sample tokens
    src = torch.randint(0, vocab_size, (2, src_seq_len)).cuda()
    tgt = torch.randint(0, vocab_size, (2, tgt_seq_len)).cuda()

    model = Transformer(
        d_model=128,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_ff=512,
        dropout=0.1,
        vocab_size=vocab_size,
        max_seq_len=max(src_seq_len, tgt_seq_len),
    ).cuda()

    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
