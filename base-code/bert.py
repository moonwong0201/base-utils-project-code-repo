import numpy as np
import torch, torch.nn as nn, math

# ===== 超参 =====
d_model = 64            # 嵌入维度
n_heads = 8             # 头数
d_ff = d_model * 4      # 前馈隐藏层（4H）

# ===== 1. 位置编码 =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        # TODO: 创建正弦位置矩阵 pe[max_len, d_model] 并 register_buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # TODO: 返回 pe[:, :x.size(1)]
        return self.pe[:, :x.size(1)]

# ===== 2. 多头注意力 =====
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # TODO: 4 个 Linear + d_k = d_model//n_heads
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_s = nn.Linear(d_model, d_model)
        self.k_s = nn.Linear(d_model, d_model)
        self.v_s = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # TODO: split→scaled_dot→concat→linear→残差
        batch_size, seq_len, _ = x.size()
        Q = self.q_s(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_s(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_s(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            print("PAD位置的注意力分数：", scores[0, 0, :, 1], scores[0, 0, :, 3])
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.linear(context)

# ===== 3. FeedForward =====
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Linear(d_model→d_ff)→GELU→Linear(d_ff→d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # TODO: 前向 + 残差
        outputs = self.linear1(x)
        outputs = nn.functional.gelu(outputs)
        outputs = self.linear2(outputs)
        return outputs

# ===== 4. Transformer Block =====
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # TODO: MultiHeadAttention + FeedForward + 2×LayerNorm
        self.multi = MultiHeadAttention(d_model, n_heads)
        self.feed = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # TODO: 残差连接：x = x + MHA(LN(x)) → x = x + FF(LN(x))
        x = x + self.multi(self.norm1(x), mask)
        x = x + self.feed(self.norm2(x))
        return x


# ===== 5. BERT 顶层 =====
class BertCore(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        # TODO: TokenEmbedding + PositionalEncoding + TransformerBlock×n_layers
        self.token_embedded = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        # TODO: 嵌入→逐块 Transformer→返回序列特征
        x = self.token_embedded(x) + self.position(x)
        for blk in self.blocks:
            x = blk(x, mask)
        return x

# ===== 6. 推理 demo =====
if __name__ == "__main__":
    bert = BertCore(vocab_size=3000, d_model=64, n_layers=2, n_heads=8, d_ff=256)
    seq = torch.LongTensor([[101, 0, 2003, 0, 102]])  # [CLS] hello world [SEP]
    mask = (seq > 0).unsqueeze(1).unsqueeze(1)  # pad mask
    out = bert(seq, mask)
    assert out.shape == (1, 5, 64), "shape error"              # (B, L, d_model)
    print("✅ BERT core 30-line forward pass OK!")