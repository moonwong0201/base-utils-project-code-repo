# -*- coding: utf-8 -*-
"""
LSTM/GRU 文本意图分类
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------- 超参数（可改） ----------
CSV_PATH = "/Users/wangyingyue/materials/大模型学习资料——八斗/第一周：课程介绍及大模型基础/Week01/Week01/dataset.csv"  # 文本\t标签，无表头
MODEL_DIR = "checkpoints"  # 模型权重保存目录
MODEL_NAME_LSTM = "lstm_model.pt"  # LSTM模型权重文件名
MODEL_NAME_GRU = "gru_model.pt"  # GRU模型权重文件名
MAX_LEN = 40  # 文本最大长度，不足补0，超出截断
TEST_SIZE = 0.2  # 测试集占比20%
RANDOM_STATE = 42  # 随机种子，保证结果可复现
BATCH_SIZE = 32  # 批次大小
EMB_DIM = 64  # 字符嵌入维度
HIDDEN_DIM = 128  # 序列模型（LSTM/GRU）隐藏层维度
NUM_EPOCHS = 10  # 训练轮数
LR = 1e-3  # 学习率
# ------------------------------------

# 设备选择：优先MPS
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")


class CharLSTMDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = labels
        self.char_to_index = char_to_index  # 字符到索引的映射表
        self.max_len = max_len  # 文本最大长度

    def __len__(self):
        # 返回数据集总样本数
        return len(self.labels)

    def __getitem__(self, idx):
        # 提取单条文本的字符索引，截断到max_len
        indices = [self.char_to_index[char] for char in self.texts[idx][:self.max_len]]
        # 补齐到max_len（不足部分用<pad>对应的0填充）
        indices += [0] * (self.max_len - len(indices))
        # 提取对应标签索引
        label_idx = self.labels[idx]
        # 返回张量格式的输入索引和标签（适配PyTorch模型输入）
        return torch.tensor(indices, dtype=torch.long), label_idx


class LSTMClassifier(nn.Module):
    """Embedding → LSTM/GRU → 隐藏状态 → 全连接分类。"""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        # 字符嵌入层：将字符索引转换为稠密向量
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)  # LSTM模型（batch_first=True表示输入格式为[batch, seq, dim]）
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)  # GRU模型（简化版LSTM，无细胞状态，计算更高效）
        # 全连接层：将GRU隐藏状态映射到分类标签空间
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 第一步：字符索引通过嵌入层转换为嵌入向量
        out = self.embedding(x)
        # 第二步：传入GRU模型获取输出和最终隐藏状态（GRU仅返回隐藏状态，无细胞状态）
        # lstm_out, (hidden_state, cell_state) = self.lstm(out)  # LSTM前向传播
        gru_out, hidden_state = self.gru(out)
        # 第三步：压缩隐藏状态的维度（去除层数维度），传入全连接层做分类
        output = self.linear(hidden_state.squeeze(0))
        return output


def build_vocab(texts):
    """返回 char_to_index, index_to_char, vocab_size。构建字符级词汇表"""
    # 初始化词汇表，<pad>对应索引0，用于文本补齐
    char_to_index = {'<pad>': 0}
    # 遍历所有文本，构建字符到索引的映射
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)  # 新字符分配递增索引

    # 构建索引到字符的反向映射（可选，用于后续调试）
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    # 返回映射表和词汇表大小
    return char_to_index, index_to_char, len(char_to_index)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch，返回平均loss。"""
    total_loss = 0.0
    model.train()  # 切换模型到训练模式（启用Dropout、BatchNorm等训练专属层）
    # 遍历训练数据加载器
    for idx, (inputs, labels) in enumerate(loader):
        # 将数据迁移到指定设备（GPU/CPU）
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零（避免上一批次梯度累积）
        optimizer.zero_grad()
        # 模型前向传播获取输出
        outputs = model(inputs)
        # 计算损失（交叉熵损失，适配分类任务）
        loss = criterion(outputs, labels)

        # 反向传播计算梯度
        loss.backward()
        # 优化器更新模型参数
        optimizer.step()

        # 累积批次损失
        total_loss += loss.item()

    # 返回本轮平均损失（总损失/批次数量）
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """在验证/测试集上返回准确率。"""
    total_correct = 0  # 累计正确预测数
    total_samples = 0  # 累计总样本数
    model.eval()  # 切换模型到评估模式（关闭Dropout、BatchNorm等训练专属层）
    with torch.no_grad():  # 关闭梯度计算，提升评估效率，避免参数更新
        for idx, (inputs, labels) in enumerate(loader):
            # 数据迁移到指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 模型前向传播获取输出
            outputs = model(inputs)
            # 取输出维度最大值对应的索引作为预测结果
            index = torch.argmax(outputs, dim=-1)
            # 计算当前批次正确预测数
            batch_correct = (index == labels).sum().item()
            # 当前批次总样本数
            batch_total = labels.size(0)

            # 累积统计数据
            total_correct += batch_correct
            total_samples += batch_total

            # 每10个批次打印一次准确率（方便监控评估过程）
            batch_acc = batch_correct / batch_total
            if idx % 10 == 0:
                print(f"第 {idx} 组准确率：{batch_acc:.4f}")

    # 计算整体准确率
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return overall_accuracy


def predict_one(text, model, char_to_index, max_len, index_to_label, device):
    """单条文本预测，返回字符串标签。"""
    model.eval()  # 切换到评估模式
    # 处理输入文本：转换为字符索引，截断+补齐
    text_index = [char_to_index.get(char, 0) for char in text[:max_len]]  # 未知字符用0填充
    text_index += [0] * (max_len - len(text_index))
    # 转换为模型输入格式（添加batch维度，迁移到指定设备）
    input_tensor = torch.tensor(text_index, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():  # 关闭梯度计算
        output = model(input_tensor)

    # 取预测结果对应的索引，转换为标签字符串
    index = torch.argmax(output, dim=-1).item()
    label = index_to_label[index]

    return label


def main():
    """总流程：读数据→划分训练测试→建词汇表→训练→评估→交互预测。"""
    # 1. 读取数据（TSV格式，无表头，分隔符为\t）
    dataset = pd.read_csv(CSV_PATH, sep='\t', header=None)
    texts = dataset[0].tolist()  # 提取文本列
    total_labels = dataset[1].tolist()  # 提取标签列
    unique_labels = list(dict.fromkeys(total_labels))  # 获取唯一标签列表（保持原有顺序）
    # 构建标签到索引的映射（用于分类任务的数值化标签）
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    # 构建索引到标签的反向映射（用于预测结果转换为字符串标签）
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    # 将原始标签转换为数值化索引
    numerical_labels = [label_to_index[label] for label in total_labels]

    # 2. 划分训练集 & 测试集（stratify保证训练集和测试集的标签分布一致）
    x_train, x_test, label_train, label_test = train_test_split(
        texts, numerical_labels,
        test_size=TEST_SIZE,
        stratify=numerical_labels
    )

    # 3. 构建字符/标签映射（基于全量文本构建词汇表）
    char_to_index, index_to_char, vocab_size = build_vocab(texts)

    # 4. 构建Dataset和DataLoader（数据加载管道，供模型训练和评估使用）
    lstm_dataset_train = CharLSTMDataset(x_train, label_train, char_to_index, MAX_LEN)
    lstm_dataset_test = CharLSTMDataset(x_test, label_test, char_to_index, MAX_LEN)

    # 训练集开启shuffle打乱数据，测试集无需打乱
    lstm_dataloader_train = DataLoader(lstm_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    lstm_dataloader_test = DataLoader(lstm_dataset_test, batch_size=BATCH_SIZE)

    # 5. 初始化模型、损失函数、优化器
    model = LSTMClassifier(vocab_size, EMB_DIM, HIDDEN_DIM, len(unique_labels)).to(device)  # 模型实例化并迁移到指定设备
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（适配多分类任务）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam优化器（收敛快，鲁棒性强）

    # 创建模型权重保存目录（不存在则自动创建）
    os.makedirs(MODEL_DIR, exist_ok=True)
    # 选择当前使用的模型权重文件（GRU）
    # model_path = os.path.join(MODEL_DIR, MODEL_NAME_LSTM)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME_GRU)

    # 检查是否存在已训练的模型权重，存在则直接加载（跳过训练）
    if os.path.exists(model_path):
        print(f"发现已存在模型文件：{model_path}，正在加载权重...")
        model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载模型权重（weights_only=True保证安全加载）
        print("模型权重加载完成")
    else:
        print(f"未发现模型文件：{model_path}，正在开始训练...")
        # 6. 训练循环（含早停可自加）
        for epoch in range(NUM_EPOCHS):
            avg_loss = train_one_epoch(model, lstm_dataloader_train, criterion, optimizer, device)
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]，平均损失：{avg_loss:.6f}")

        # 训练完成后保存模型权重
        torch.save(model.state_dict(), model_path)
        print(f"模型训练完成，权重已保存至：{model_path}")

    # 7. 最终评估：在测试集上评估模型整体准确率
    overall_acc = evaluate(model, lstm_dataloader_test, device)
    print(f"\n测试集整体准确率：{overall_acc:.4f}\n")

    # 8. 交互预测：对预设测试用例进行逐一预测并输出结果
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
        label = predict_one(text, model, char_to_index, MAX_LEN, index_to_label, device)
        print(f"待分类文本：{text}")
        print(f"预测结果：{label}\n")


if __name__ == "__main__":
    # 程序入口：执行main函数
    main()