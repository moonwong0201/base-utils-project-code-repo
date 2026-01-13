# -*- coding: utf-8 -*-
"""
字符级词袋（Char-level Bag of Words） + 两层全连接
用于中文意图分类的基础 Demo（单文件版）

核心思想：
1. 每个句子 → 字符级 BoW 向量（不考虑顺序，只统计出现次数）
2. BoW 向量 → 两层全连接网络 → 意图类别
"""

import argparse
import os
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ================== 设备选择 ==================
# 优先使用 Apple MPS（Mac），否则使用 CPU
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用设备: {device}")

# ================== 超参数 ==================
CSV_PATH = "dataset.csv"
MAX_LEN = 40          # 每条文本最多使用的字符数
HIDDEN_DIM = 32       # 隐藏层维度
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
MODEL_PATH = "checkpoints/bow_model.pt"
# ============================================


class CharBoWDataset(Dataset):
    """
    字符级 BoW 数据集

    输入：
    - 原始文本列表
    - 标签
    - 字符到索引的映射表

    输出：
    - (BoW 向量, label)
    """
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = labels.to(device)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size

        # 在初始化时一次性构建所有 BoW 向量
        # 避免每个 batch 动态计算，提升训练效率
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        """
        将每条文本转换为 BoW 向量
        输出形状：(N, vocab_size)
        """
        tokenized_texts = []
        for text in self.texts:
            # 字符 → 索引，超过 MAX_LEN 的截断
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 不足 MAX_LEN 的补 0（pad）
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            # BoW：统计每个字符出现的次数
            bow_vector = torch.zeros(self.vocab_size, device=device)
            for index in text_indices:
                if index != 0:  # 忽略 pad
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)

        return torch.stack(bow_vectors).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    """
    两层全连接分类器

    结构：
    BoW → Linear → ReLU → Dropout → Linear → logits
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 防止 BoW 场景下过拟合
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


def build_vocab(texts):
    """
    根据训练文本构建字符级词表

    返回：
    - char_to_index：字符 → 索引
    - index_to_char：索引 → 字符
    - vocab_size：词表大小
    """
    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    index_to_char = {i: char for char, i in char_to_index.items()}
    vocab_size = len(char_to_index)
    return char_to_index, index_to_char, vocab_size


def train():
    """
    训练流程：
    1. 读取数据
    2. 构建词表 & 标签映射
    3. 划分训练 / 验证集
    4. 训练模型并保存
    """
    dataset = pd.read_csv(CSV_PATH, sep='\t', header=None)
    texts = dataset[0].tolist()
    labels = dataset[1].tolist()

    # 标签数值化
    labels_to_index = {label: i for i, label in enumerate(set(labels))}
    numerical_labels = torch.tensor(
        [labels_to_index[label] for label in labels],
        dtype=torch.long
    )

    # 构建字符词表
    char_to_index, index_to_char, vocab_size = build_vocab(texts)

    # 划分训练集 / 验证集
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts,
        numerical_labels,
        test_size=0.2,
        random_state=42,
        stratify=numerical_labels
    )

    train_dataset = CharBoWDataset(
        texts_train, labels_train, char_to_index, MAX_LEN, vocab_size
    )
    val_dataset = CharBoWDataset(
        texts_val, labels_val, char_to_index, MAX_LEN, vocab_size
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleClassifier(vocab_size, HIDDEN_DIM, len(labels_to_index)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    print("=" * 50)
    print("开始训练模型...")

    for epoch in range(EPOCHS):
        # ===== 训练阶段 =====
        model.train()
        total_loss = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss = {total_loss / len(train_dataloader):.6f}")

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}, Val Loss = {val_loss / len(val_dataloader):.6f}\n")

    # 保存最终模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'char_to_index': char_to_index,
        'label_to_index': labels_to_index,
        'vocab_size': vocab_size
    }, MODEL_PATH)

    print("模型已保存至", MODEL_PATH)


def predict():
    """
    加载模型并对固定测试样例进行预测
    """
    base_model = torch.load(MODEL_PATH, map_location=device)

    char_to_index = base_model['char_to_index']
    label_to_index = base_model['label_to_index']
    index_to_label = {i: l for l, i in label_to_index.items()}
    vocab_size = base_model['vocab_size']

    model = SimpleClassifier(vocab_size, HIDDEN_DIM, len(label_to_index))
    model.load_state_dict(base_model['model'])
    model.to(device)
    model.eval()

    test_cases = [
        '查询下周三我的日程安排是什么',
        '帮我播放《狂飙》第15集',
        '今天天气不错，适合出门散步',
        '给我播放张敬轩的歌曲《春秋》',
        '帮我打开客厅的空调，设置温度26度',
        '帮我播放湖南卫视的《快乐大本营》最新一期',
        '收听FM93.8交通广播的实时路况',
        '查询明天上海的天气情况，是否有雨',
        '把明天早上7点的闹钟修改为7点30分',
        '播放《明朝那些事儿》的有声小说第10章',
        '帮我找到电影《流浪地球2》并播放',
        '查询从北京到上海的高铁票，明天下午出发的'
    ]

    for text in test_cases:
        tokenized = [char_to_index.get(char, 0) for char in text[:MAX_LEN]]
        tokenized += [0] * (MAX_LEN - len(tokenized))

        bow_vector = torch.zeros(vocab_size, device=device)
        for index in tokenized:
            if index != 0:
                bow_vector[index] += 1

        with torch.no_grad():
            outputs = model(bow_vector.unsqueeze(0))
            pred = torch.argmax(outputs, dim=-1).item()

        print(f"文本：{text}")
        print(f"预测意图：{index_to_label[pred]}\n")


def main():
    if os.path.exists(MODEL_PATH):
        predict()
    else:
        train()
        predict()


if __name__ == "__main__":
    main()
