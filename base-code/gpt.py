import torch, torch.nn as nn, math

# ===== 超参 =====
d_model = 64        # 嵌入维度
n_heads = 8         # 头数
d_ff = d_model * 4  # 前馈隐藏层
n_layers = 2        # 解码器层数


# ===== 1. 位置编码 =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# ===== 2. 多头注意力 =====
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // self.n_heads
        self.q_s = nn.Linear(d_model, d_model)
        self.k_s = nn.Linear(d_model, d_model)
        self.v_s = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, residual = x.size(0), x
        Q = self.q_s(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_s(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_s(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask == 1, -1e9)

        attn_weight = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weight, V).contiguous().view(batch_size, -1, self.d_model)
        return self.linear(context) + residual


# ===== 3. FeedForward =====
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        output = self.linear1(x)
        output = nn.functional.gelu(output)
        output = self.linear2(output)
        return output

# ===== 4. Transformer Block =====
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.feed = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x)
        x = x + self.attn(x, mask)
        return x + self.feed(self.norm2(x))

# ===== 5. GPT 顶层 =====
class GPTCore(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=128):
        super().__init__()
        self.embedded = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedded(x) + self.position(x)
        for blk in self.blocks:
            x = blk(x, mask)
        output = self.linear(x)
        return output


# ===== 6. 推理 demo =====
if __name__ == "__main__":
    vocab_size = 3000
    model = GPTCore(vocab_size, d_model=64, n_layers=2, n_heads=8, d_ff=256)
    seq = torch.LongTensor([[1, 2, 3, 4, 5]])  # [sos, 机, 器, 学, 习]
    mask = torch.triu(torch.ones(seq.size(1), seq.size(1)), diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(0)
    logits = model(seq, mask)
    assert logits.shape == (1, 5, vocab_size), "shape error"
    print("✅ GPT core 30-line forward pass OK!")
