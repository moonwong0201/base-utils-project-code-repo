# -*- coding: utf-8 -*-
"""
基于 Transformer 文本意图分类
"""

import os
import math
import jieba
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ========== 全局超参数配置 ==========
CSV_PATH = "dataset.csv"
MODEL_DIR = "checkpoints"
MODEL_NAME = "transformer_cls.pt"
MAX_LEN = 40
BATCH_SIZE = 32
D_MODEL = 128
N_HEAD = 8
ENC_LAYERS = 2
DIM_FF = 128
DROPOUT = 0.2
NUM_EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ========== 1. 数据加载与预处理 ==========
def DataPrepare(texts, unique_labels):
    """
    数据预处理：构建词语/标签与索引的映射，完成中文文本分词
    Args:
        texts: 原始文本列表
        unique_labels: 去重后的标签列表
    Returns:
        word_to_index: 词语到索引的映射字典
        label_to_index: 标签到索引的映射字典
        vocab_size: 词汇表总大小（包含<pad>和<unk>）
        texts_tokenized: 分词后的文本列表
    """
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    word_to_index = {'<pad>': 0, '<unk>': 1}
    texts_tokenized = [jieba.lcut(text) for text in texts]
    for text in texts_tokenized:
        for token in text:
            if token not in word_to_index:
                word_to_index[token] = len(word_to_index)

    vocab_size = len(word_to_index)

    return word_to_index, label_to_index, vocab_size, texts_tokenized


# ========== 2. Dataset ==========
class TextDataset(Dataset):
    """
    自定义PyTorch Dataset，用于处理文本数据并转换为模型可输入的张量格式
    """
    def __init__(self, tokenized_texts, labels, word_to_index, max_len):
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_len = max_len

    def __len__(self):
        """返回数据集的总样本数量（PyTorch Dataset 必需实现）"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本数据，完成序列截断/填充与张量转换（PyTorch Dataset 必需实现）
        Args:
            idx: 样本索引
        Returns:
            tokenized_index: 处理后的文本序列张量
            label: 处理后的标签张量
        """
        tokenized_index = [self.word_to_index.get(word, 1) for word in self.tokenized_texts[idx][:self.max_len]]
        tokenized_index += [0] * (self.max_len - len(tokenized_index))
        tokenized_index = torch.tensor(tokenized_index, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tokenized_index, label


# ========== 3. 位置编码 ==========
class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码：为Transformer提供序列位置信息，弥补其无时序建模能力的缺陷
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """前向传播：将位置编码与词嵌入张量相加，注入位置信息"""
        return x + self.pe[:, :x.size(1), :]


# ========== 4. Transformer 分类模型 ==========
class TransformerClassifier(nn.Module):
    """
    Transformer文本分类模型：词嵌入 -> 位置编码 -> Transformer编码器 -> 全局平均池化 -> 分类
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feed = nn.Linear(d_model, num_classes)

    def forward(self, src):
        """
        模型前向传播：完成文本特征提取与分类预测
        Args:
            src: 输入的文本序列索引张量
        Returns:
            out: 标签空间的预测结果张量
        """
        out = self.embedding(src) * math.sqrt(self.d_model)
        out = self.position(out)
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)
        out = self.feed(out)
        return out


# ========== 5. 训练 & 评估 ==========
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_path):
    """
    模型训练函数：批量训练模型，每轮验证并保存最优准确率模型
    Args:
        model: 待训练的模型实例
        train_loader: 训练集数据加载器
        val_loader: 验证集（测试集）数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备（MPS/CPU/GPU）
        epochs: 训练轮数
        model_path: 最优模型权重保存路径
    Returns:
        model: 训练完成的最优模型
        best_acc: 训练过程中的最优验证准确率
    """
    best_acc = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

        train_avg_loss = train_total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch + 1}, Train Loss = {train_total_loss / len(train_loader):.6f}")
        if val_acc > best_acc:
            best_acc = val_acc
            # 深拷贝最优模型权重，避免后续训练覆盖
            best_model_weights = model.state_dict().copy()
            print(f"\n发现最优模型，当前最优测试集准确率：{best_acc:.4f}")

    # 加载最优模型权重并保存到本地
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        torch.save(best_model_weights, model_path)
        print(f"\n训练完成，最优模型已保存至：{model_path}，最优测试集准确率：{best_acc:.4f}")

    return model, best_acc


def evaluate(model, data_loader, device):
    """
    模型评估函数：在无梯度环境下计算模型分类准确率
    Args:
        model: 待评估的模型实例
        data_loader: 验证集/测试集数据加载器
        device: 评估设备（MPS/CPU/GPU）
    Returns:
        overall_correct_rate: 模型整体分类准确率
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            label_index = torch.argmax(outputs, dim=-1)
            correct_sum = (label_index == labels).sum().item()
            batch_size = labels.size(0)

            total_samples += batch_size
            total_correct += correct_sum

            if idx % 15 == 0:
                correct_rate = correct_sum / batch_size
                print(f"第 {idx + 1} 组准确率为：{correct_rate}")

    overall_correct_rate = total_correct / total_samples
    return overall_correct_rate


# ========== 6. 预测接口 ==========
def predict(text, model, word_to_index, max_len, index_to_label, device):
    """
    单条文本预测接口：接收原始中文文本，返回对应的意图标签
    Args:
        text: 原始中文输入文本
        model: 训练完成的最优模型实例
        word_to_index: 词语到索引的映射字典
        max_len: 文本序列最大长度
        index_to_label: 索引到标签的映射字典
        device: 预测设备（MPS/CPU/GPU）
    Returns:
        label: 文本对应的意图标签
    """
    tokenized_text = jieba.lcut(text)
    text_index = [word_to_index.get(word, 1) for word in tokenized_text[:max_len]]
    text_index += [0] * (max_len - len(text_index))
    text_index = torch.tensor(text_index, dtype=torch.long).unsqueeze(0).to(device)
    text_index = text_index.to(device)
    with torch.no_grad():
        outputs = model(text_index)
        index = torch.argmax(outputs, dim=-1).item()
        label = index_to_label[index]
    return label


# ========== 7. main ==========
def main():
    """
    程序入口函数：完成数据加载、模型初始化、训练/加载、预测验证的全流程
    """
    # 加载原始数据集
    dataset = pd.read_csv(CSV_PATH, sep='\t', header=None)
    texts = dataset[0].tolist()
    labels = dataset[1].tolist()
    unique_labels = list(dict.fromkeys(labels))
    word_to_index, label_to_index, vocab_size, texts_tokenized = DataPrepare(texts, unique_labels)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    numerical_label = [label_to_index[label] for label in labels]
    x_train, x_test, label_train, label_test = train_test_split(
        texts_tokenized, numerical_label,
        test_size=0.2,
        random_state=42,
        stratify=numerical_label
    )

    dataset_train = TextDataset(x_train, label_train, word_to_index, MAX_LEN)
    dataset_test = TextDataset(x_test, label_test, word_to_index, MAX_LEN)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE)

    num_classes = len(unique_labels)
    model = TransformerClassifier(vocab_size, D_MODEL, N_HEAD, ENC_LAYERS, DIM_FF, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # 创建模型权重保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    # 检查是否存在已训练模型，存在则加载，否则训练
    if os.path.exists(model_path):
        print(f"发现已存在模型文件：{model_path}，正在加载权重...")
        model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载模型权重（weights_only=True保证安全加载）
        print("模型权重加载完成")
    else:
        print(f"\n未发现模型文件：{model_path}，正在开始训练...")
        model, best_acc = train(model, dataloader_train, dataloader_test, criterion, optimizer, DEVICE, NUM_EPOCHS, model_path)
        print(f"\n测试集整体准确率：{best_acc:.4f}\n")

    # 测试用例验证预测效果
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
        label = predict(text, model, word_to_index, MAX_LEN, index_to_label, DEVICE)
        print(f"待分类文本：{text}")
        print(f"预测结果：{label}\n")


if __name__ == "__main__":
    main()
