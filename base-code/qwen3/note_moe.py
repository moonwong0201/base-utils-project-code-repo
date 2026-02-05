import torch
import torch.nn as nn


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    positions = torch.arange(context_length, dtype=dtype)

    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)
  
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]  # 每个token选8个专家
        self.num_experts = cfg["num_experts"]  # 总128个专家
        self.emb_dim = cfg["emb_dim"]  # 输入输出维度

        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        # 创建多个独立的专家（每个专家都是一个 SwiGLU 结构的 FFN）
        self.fc1 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_hidden_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc2 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_hidden_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc3 = nn.ModuleList([nn.Linear(cfg["moe_hidden_dim"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])

    def forward(self, x):
        # 门控网络计算每个专家的得分（shape: (b, seq_len, 128)）
        scores = self.gate(x)  # (b, seq_len, num_experts)

        # 选得分最高的8个专家，得到：
        # topk_scores: 选中专家的得分（shape: (b, seq_len, 8)）
        # topk_indices: 选中专家的ID（shape: (b, seq_len, 8)）
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        # 对选中专家的得分做softmax，得到权重（shape: (b, seq_len, 8)）
        topk_probs = torch.softmax(topk_scores, dim=-1)

        # 展平维度，方便批量处理（b, seq_len, emb_dim）→ (b*seq_len, emb_dim)
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        # 展平选中的专家ID和权重（方便后续按专家分组计算）
        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)  # (b*seq_len, 8)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)      # (b*seq_len, 8)

        # 找出所有被选中的唯一专家（避免重复计算）
        unique_experts = torch.unique(topk_indices_flat)

        # 遍历每个被选中的专家，计算其输出并加权
        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())  # 专家ID（如0,1,2...）

            # 找到哪些token选了这个专家（mask: (b*seq_len, 8)）
            mask = topk_indices_flat == expert_id
            if not mask.any():  # 无token选这个专家，跳过
                continue

            # 判断 “每个 token 是否选了当前专家”（只要该 token 的 2 个选中专家里有当前专家，就是 True）
            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)  # 筛选出选了该专家的token（shape: (num_selected_tokens,)）
            if selected_idx.numel() == 0:
                continue

            # 取出这些token的输入，传入该专家计算
            expert_input = x_flat.index_select(0, selected_idx)
            # 专家的SwiGLU计算（和普通FFN逻辑一致）
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            expert_out = self.fc3[expert_id](hidden)

            # 取出这些token对该专家的权重
            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)

            # 加权后累加到输出中（核心：多个专家的输出加权求和）
            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        return out_flat.reshape(batch, seq_len, self.emb_dim)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )

        if cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)

        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x



class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(
            # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


QWEN3_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 1024,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "hidden_dim": 3072,
    "head_dim": 64,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000_000.0,
    "dtype": torch.float32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_hidden_dim": 512,
}

# QWEN3_CONFIG = {
#     "vocab_size": 151_936,
#     "context_length": 262_144,  # 上下文长度（上下文窗口）：模型能处理的最大 token 序列长度
#     "emb_dim": 2048,
#     "n_heads": 32,  # Query（Q）头数：多头注意力中 Q 的数量
#     "n_layers": 48,
#     "hidden_dim": 3072,
#     "head_dim": 128,  # 单个注意力头的维度：每个 Q/K/V 头的向量维度
#     "qk_norm": True,  # 是否对 Q/K 做 RMSNorm 归一化
#     "n_kv_groups": 4,  # KV 分组数（GQA 核心）：将 32 个 Q 头分成 4 组，每组共享 1 个 K/V 头
#     "rope_base": 10_000_000.0,  # RoPE（旋转位置编码）的频率基数
#     "dtype": torch.bfloat16,
#     "num_experts": 128,  # 总专家数：FFN 层的独立专家数量（每个专家都是一个 SwiGLU 结构的 FFN）
#     "num_experts_per_tok": 8,  # 每个 token 选择的专家数：门控网络为每个 token 挑选得分最高的 8 个专家参与计算
#     "moe_hidden_dim": 768,  # 每个专家 FFN 的中间层维度
# }


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)

torch.manual_seed(123)

with device:
    model = Qwen3Model(QWEN3_CONFIG)



model(torch.tensor([1, 2, 3]).unsqueeze(0).to(device))

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
