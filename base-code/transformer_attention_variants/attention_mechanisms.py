"""
实现了 Transformer 架构中三种经典的注意力机制：
MultiHeadAttention：每个 Query/Key/Value 都有相同数量的 head；
MultiQueryAttention：所有 Query head 共享一组 Key/Value head，减少显存占用；
GroupQueryAttention：MQA 和 MHA 的折中，将 Query head 分组，每组共享一组 Key/Value head。
"""
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()  # 继承nn.Module的初始化
        self.nums_head = nums_head  # 注意力头的数量（如8）

        self.head_dim = hidden_dim // nums_head  # 每个head的维度（如128//8=16）
        self.hidden_dim = hidden_dim  # 隐藏层总维度（如128）

        # 线性投影层：将输入hidden_dim映射为Q/K/V（维度不变）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # Q投影层
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)  # K
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)  # V

        self.att_dropout = nn.Dropout(0.1)   # 注意力权重的dropout（防止过拟合）

        # 输出投影层：将拼接后的多头结果映射回hidden_dim
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        # 需要在 mask 之前 masked_fill
        # X shape is (batch, seq, hidden_dim)
        # attention_mask shape is (batch, seq)

        batch_size, seq_len, _ = X.size()

        # 生成Q/K/V：通过线性投影层
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # shape 变成 （batch_size, num_head, seq_len, head_dim）
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        # 主意这里需要用 head_dim，而不是 hidden_dim
        attention_weight = (q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim))

        # print("MHA 掩码：\n", type(attention_mask))

        # 应用注意力掩码：把mask=0的位置设为极小值（softmax后权重为0）
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-1e20")  # 极小值（比-inf更稳定）
            )

        # 计算softmax：对最后一维（seq）归一化，得到注意力权重
        attention_weight = torch.softmax(attention_weight, dim=3)  # dim=3对应seq维度
        # print("MHA 注意力权重：\n", attention_weight)

        # 注意力权重dropout
        attention_weight = self.att_dropout(attention_weight)
        # 计算注意力输出：权重 @ V
        output_mid = attention_weight @ v_state

        # 重新变成 (batch, seq_len, num_head, head_dim)
        # 这里的 contiguous() 是相当于返回一个连续内存的 tensor，一般用了 permute/tranpose 都要这么操作
        # 如果后面用 Reshape 就可以不用这个 contiguous()，因为 view 只能在连续内存中操作
        output_mid = output_mid.transpose(1, 2).contiguous()

        # view拼接：(batch, seq, num_head*head_dim) = (batch, seq, hidden_dim)
        output = output_mid.view(batch_size, seq_len, -1)  # -1自动计算为hidden_dim
        # 输出投影：线性层映射回hidden_dim（融合多头信息）
        output = self.o_proj(output)
        return output


attention_mask = (
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [1, 0],
        ]
    )
    .unsqueeze(1)  # 增加维度：(3,1,2) → 对应num_head维度
    .unsqueeze(2)  # 增加维度：(3,1,1,2) → 对应seq维度
    .expand(3, 8, 2, 2)  # 扩展维度：(3,8,2,2)（适配8个head，seq_len=2）
)

x = torch.rand(3, 2, 128)
net = MultiHeadAttention(128, 8)
print("MHA 前向传播形状：", net(x, attention_mask).shape)


class MultiQueryAttention(nn.Module):

    def __init__(self, hidden_dim, nums_head):
        super().__init__()
        assert hidden_dim % nums_head == 0  # hidden_dim 必须能被 nums_head 整除

        self.hidden_dim = hidden_dim  # 总隐藏维度
        self.nums_head = nums_head    # Query head数量
        # MQA核心：Key/Value head数量固定为1（所有Query head共享一组K/V）
        self.nums_key_value_head = 1
        self.head_dim = hidden_dim // nums_head  # 每个head的维度

        # 线性投影层：Q投影到num_head*head_dim，K/V投影到1*head_dim
        self.q_proj = nn.Linear(hidden_dim, self.nums_head * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)

        self.att_dropout = nn.Dropout(0.1)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        batch_size, seq_len, _ = X.size()

        q = self.q_proj(X)  # (b, seq, hidden_dim)
        k = self.k_proj(X)  # (b, seq, 1 * head_dim)
        v = self.v_proj(X)  # (b, seq, 1 * head_dim)

        q = q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)

        # MQA核心：将K/V复制到和Q相同的head数量（共享K/V）(b, nums_head, seq, head_dim)
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_score = attention_score.masked_fill(
                attention_mask == 0, float("-inf")
            )

        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_dropout(attention_weight)

        output_mid = attention_weight @ v

        # 拼接多头结果（和标准多头一致）
        output_mid = output_mid.transpose(1, 2).contiguous()
        output_mid = output_mid.view(batch_size, seq_len, -1)
        final_output = self.o_proj(output_mid)

        return final_output

x = torch.rand(3, 2, 128)
nums_head = 8
net_mqa = MultiQueryAttention(128, nums_head)
output_mqa = net_mqa(x, attention_mask)
print("MQA 前向传播形状：", output_mqa.shape)


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0  # 可以整除
        assert nums_head % nums_key_value_head == 0  # N 个 query head 为一组

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        # 线性投影层：Q投影到num_head*head_dim，K/V投影到num_key_value_head*head_dim
        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)  # out feature_size (nums_head * head_dim)
        # k v out shape (nums_key_value_head * head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim)  # input_size nums_head * head_dim

    def forward(self, X, attention_mask=None):
        # X shape (batch, seq, hidden_dim)
        batch_size, seq, _ = X.size()

        # qkv projection
        q = self.q_proj(X)  # （batch, seq, hidden_dim)
        k = self.k_proj(X)
        v = self.v_proj(X)

        # attention_weight 目标shape 是 (batch, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # 关注: nums_head 和 nums_key_value_head 的关系
        q = q.transpose(1, 2)  # (b, nums_head, seq, head_dim)
        k = k.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)
        v = v.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)

        # k v repeat； （广播操作）
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_score = attention_score.masked_fill(
                attention_mask == 0, float("-inf")
            )

        attention_weight = torch.softmax(attention_score, dim=-1)

        output = attention_weight @ v  # (b, nums_head, seq, head_dim)

        # output projection 变成 (b, seq, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        final_output = self.o_proj(output.view(batch_size, seq, -1))

        return final_output


# 测试
x = torch.rand(3, 2, 128)
net = GroupQueryAttention(128, 8, 4)
print("GQA 前向传播形状：", net(x, attention_mask).shape)
