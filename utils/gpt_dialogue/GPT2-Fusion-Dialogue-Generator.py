# -*- coding: utf-8 -*-
"""
GPT-2 中文对话微调（融合式版）
"""

import os

# MPS 显存相关环境变量
# 作用：降低 PyTorch 在 Apple MPS 上的显存保留策略，避免“假性 OOM”
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE_MEMORY_POOL"] = "1"
os.environ["PYTORCH_MPS_FAST_MEMORY"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# 全局参数
# 自动选择设备：MPS / CUDA / CPU
DEVICE = "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else \
    "cuda" if torch.cuda.is_available() else "cpu"

# 本地 GPT-2 模型路径
MODEL_NAME = "models/openai-community/gpt2"

# 对话数据路径（User: / AI: 格式）
DATA_FILE_PATH = "chat.txt"

# 微调后模型保存路径
MODEL_SAVE_PATH = "best_gpt2_chat.pth"

# 训练超参数
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 0.01

# 生成相关参数
GEN_MAX_LEN = 40
BEAM_WIDTH = 4  # num_beams > 1 即 Beam Search


# 用于显式区分 User 和 AI
SEP_TOKEN = "|"


# tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# 如果分隔符不在词表中，则手动加入
if SEP_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([SEP_TOKEN])

# GPT-2 原生没有 pad_token，这里复用 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 加载 GPT-2 模型
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# 因为 tokenizer 加了新 token，模型 embedding 大小要同步调整
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)

# 分隔符对应的 token id
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(SEP_TOKEN)


# Dataset
class ChatDataset(Dataset):
    """
    数据集设计：
    - 输入格式：User + SEP + AI + EOS
    """
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = self.load(file_path)

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        user = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 读取 User 输入
            if line.startswith("User:"):
                user = self.tokenizer.encode(line[6:], add_special_tokens=False)

            # 读取 AI 回复，并与 User 融合成一个训练样本
            elif line.startswith("AI:") and user is not None:
                ai = self.tokenizer.encode(line[4:], add_special_tokens=False)

                # 核心融合格式
                seq = user + [SEP_TOKEN_ID] + ai + [tokenizer.eos_token_id]
                data.append(torch.tensor(seq, dtype=torch.long))
                user = None

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# collate
def collate_fn(batch):
    """
    - 对不同长度的序列进行 padding
    """
    max_len = max(len(x) for x in batch)
    padded = torch.full(
        (len(batch), max_len),
        tokenizer.pad_token_id,
        dtype=torch.long
    )
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# 构造 labels（只监督 AI）
def build_labels(input_ids):
    """
    核心思想：
    - GPT 是自回归模型，但我们不希望它“学会预测 User”
    - User + SEP 部分的 loss 全部 mask 掉
    """
    labels = input_ids.clone()

    for i in range(input_ids.size(0)):
        sep_pos = (input_ids[i] == SEP_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            # SEP 之前（包括 SEP）全部忽略
            labels[i, :sep_pos[0] + 1] = -100

    return labels


# training
def train(model, loader):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ignore_index = -100 对应上面 mask 的 label
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(DEVICE)

            labels = build_labels(batch)

            # 传入 labels，HF 会自动做 shift + loss
            outputs = model(input_ids=batch, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)


# generation
def generate(model, text):
    """
    - 让模型自回归生成 SEP 后面的内容（AI）
    """
    model.eval()

    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids += [SEP_TOKEN_ID]
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)

    # 明确提供 attention_mask，避免 generate warning
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        # .generate 能够直接输出预测的字的索引
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.size(1) + GEN_MAX_LEN,
            num_beams=BEAM_WIDTH,
            eos_token_id=tokenizer.eos_token_id
        )

    out = out[0]

    # 只解码 SEP 后面的内容，作为 AI 回复
    sep_idx = (out == SEP_TOKEN_ID).nonzero(as_tuple=True)[0][0] + 1
    return tokenizer.decode(out[sep_idx:], skip_special_tokens=True)

# main
def main():
    dataset = ChatDataset(DATA_FILE_PATH, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 如果已有训练好的模型，直接加载
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        train(model, loader)

    tests = [
        "hi, how are you?",
        "what is the weather like today?",
        "can you recommend a good book?"
    ]

    for t in tests:
        print("User:", t)
        print("AI:", generate(model, t))
        print()


if __name__ == "__main__":
    main()
