import torch
import torch.nn as nn
from importlib.metadata import version


def compute_rope_params(head_dim, theta_base=10_000, context_length=40960, dtype=torch.float32):
    """
    预计算RoPE（旋转位置编码）所需的所有位置、所有维度的cos/sin值
    :param head_dim: 单个注意力头的维度
    :param theta_base: RoPE的频率基数
    :param context_length: 模型支持的最大序列长度
    :param dtype: 计算精度
    :return: cos: [context_length, head_dim] 每个位置+维度的余弦值
             sin: [context_length, head_dim] 每个位置+维度的正弦值
    """
    # 断言：head_dim必须是偶数，否则无法拆分为实部/虚部进行旋转计算
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 计算RoPE的基础逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 生成所有位置索引
    positions = torch.arange(context_length, dtype=dtype)

    # 计算每个位置×每个维度组的旋转角度
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

    # 复制角度到偶数/奇数维度
    # 拼接后 → [context_length, head_dim]，匹配注意力头的维度
    angles = torch.cat([angles, angles], dim=1)

    # 计算每个位置+维度的cos/sin值
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    将RoPE旋转位置编码应用到Q/K向量上
    :param x: 待旋转的张量（Q/K），形状: (batch_size, num_heads, seq_len, head_dim)
    :param cos: 预计算的余弦矩阵，形状: (context_length, head_dim)
    :param sin: 预计算的正弦矩阵，形状: (context_length, head_dim)
    :return: 融入位置信息后的张量，形状与输入x一致
    """
    # 获取输入张量的维度信息
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0

    # 将注意力头的维度拆分为两半，模拟复数的实部和虚部
    x1 = x[..., : head_dim // 2]  
    x2 = x[..., head_dim // 2:]  

    # 调整cos/sin的形状，适配输入x的维度
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 生成旋转辅助张量（对应复数旋转90度的正交变换）
    rotated = torch.cat((-x2, x1), dim=-1)

    # 核心旋转计算
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """
    RMSNorm归一化层
    """

    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps 
        self.qwen3_compatible = qwen3_compatible  # 是否兼容Qwen3的计算逻辑
        # 可训练的缩放参数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可选的偏移参数
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # RMS归一化
        norm_x = x * torch.rsqrt(variance + self.eps)

        # 应用可训练的缩放参数
        norm_x = norm_x * self.scale

        # 可选应用偏移参数
        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力
    """

    def __init__(
            self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads  # Q头总数
        self.num_kv_groups = num_kv_groups  # KV分组数
        self.group_size = num_heads // num_kv_groups  # 每组包含的Q头数

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim  # 单个注意力头的维度
        self.d_out = num_heads * head_dim  # 所有Q头的总输出维度

        # Q/K/V投影层
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype) 
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype) 
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)  

        # 注意力输出投影层
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # Q/K归一化
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        batch, num_tokens, _ = x.shape

        # Q/K/V投影
        queries = self.W_query(x)  # [batch, num_tokens, num_heads*head_dim]
        keys = self.W_key(x)  # [batch, num_tokens, num_kv_groups*head_dim]
        values = self.W_value(x)  # [batch, num_tokens, num_kv_groups*head_dim]

        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Q/K归一化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # 应用RoPE旋转位置编码
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # KV头扩展，匹配Q头数量
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 应用因果掩码
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5),
                                     dim=-1)  # 注：原代码dim=1是笔误，修正为dim=-1（不影响你的代码，仅注释说明）

        # 计算注意力输出并恢复维度顺序
        context = (attn_weights @ values).transpose(1, 2).reshape(batch, num_tokens, self.d_out)

        # 输出投影
        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    前馈网络（FFN）：Qwen3的SwiGLU变体
    silu激活+残差连接，比传统FFN更高效
    """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)  

    def forward(self, x):
        # 两层线性投影
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)

        # silu激活 + 残差支路
        x = nn.functional.silu(x_fc1) + x_fc2

        # 输出投影
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """
    Transformer基础块：结构：预归一化 + 注意力 + 残差连接 + 预归一化 + FFN + 残差连接
    """

    def __init__(self, cfg):
        super().__init__()
        # 分组查询注意力层
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        # 前馈网络层
        self.ff = FeedForward(cfg)
        # 两层RMSNorm
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        shortcut = x 
        x = self.norm1(x)  # 预归一化
        x = self.att(x, mask, cos, sin)  # 注意力计算
        x = x + shortcut  # 残差连接

        shortcut = x  
        x = self.norm2(x) 
        x = self.ff(x) 
        x = x + shortcut 

        return x


class Qwen3Model(nn.Module):
    """
    完整Transformer架构：词嵌入 + 多层TransformerBlock + 最终归一化 + 输出头
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # 保存模型配置
        self.tok_emb = nn.Embedding(self.cfg["vocab_size"], self.cfg["emb_dim"], dtype=self.cfg["dtype"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(self.cfg) for _ in range(self.cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        if self.cfg["head_dim"] is None:
            head_dim = self.cfg["emb_dim"] // self.cfg["n_heads"]
        else:
            head_dim = self.cfg["head_dim"]

        self.cos, self.sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=self.cfg["rope_base"],
            context_length=self.cfg["context_length"]
        )

        self.register_buffer("cos", self.cos, persistent=False)
        self.register_buffer("sin", self.sin, persistent=False)

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]  # 当前序列长度
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1  # 上三角矩阵
        )

        # 逐层执行TransformerBlock
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        return logits


# Qwen3模型配置
QWEN3_CONFIG = {
    "vocab_size": 151_936,     # 词表大小
    "context_length": 40_960,  # 模型支持的最大序列长度
    "emb_dim": 1024,           # 嵌入维度
    "n_heads": 16,             # Q头总数
    "n_layers": 28,            # TransformerBlock堆叠层数
    "hidden_dim": 3072,        # FFN中间层维度
    "head_dim": 128,           # 单个注意力头的维度
    "qk_norm": True,           # 是否对Q/K做RMSNorm
    "n_kv_groups": 8,          # KV分组数
    "rope_base": 1_000_000.0,  # RoPE频率基数
    "dtype": torch.bfloat16,   # 模型参数类型
}

# 初始化Qwen3模型
model = Qwen3Model(QWEN3_CONFIG)

# 统计模型的总参数数量（包括所有可训练参数）
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
