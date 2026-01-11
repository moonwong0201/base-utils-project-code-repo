import os.path
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# ===== 1. 超参 =====
# 本地存放bert-base-chinese预训练模型的目录，包含词汇表、模型权重等核心文件
model_path = "bert-base-chinese"
# 文本分类任务的总类别数量
num_class = 12
# 每次训练/验证的样本批次大小，即一次传入模型的样本数
batch_size = 16
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
# 初始化BERT分词器：用于将中文文本转换为模型可识别的输入格式（input_ids/attention_mask等）
tokenizer = BertTokenizer.from_pretrained(model_path)
# 初始化BERT序列分类模型：基于预训练模型添加分类头，适配指定类别数的分类任务
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_class)
model.to(device)
# 初始化AdamW优化器：BERT微调专用优化器，学习率2e-5为行业常用最优值
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ===== 2. 数据集 =====
# 加载CSV格式数据集
dataset = pd.read_csv(
    "dataset.csv",
    sep='\t',
    header=None
)
print(set(dataset[1]))
# 初始化标签编码器：用于将文本格式标签转换为数字格式标签，适配模型训练输入要求
lbl = LabelEncoder()
# 拟合标签编码器：学习数据集中所有唯一标签的映射关系，生成标签→数字的对应规则
lbl.fit(dataset[1].values)

# 数据集拆分：8:2比例拆分为训练集和测试集，保证类别分布均衡
x_train, x_test, label_train, label_test = train_test_split(
    list(dataset[0].values[:500]),  
    lbl.transform(dataset[1].values[:500]), 
    test_size=0.2, 
    stratify=dataset[1][:500].values  # 分层拆分，保持训练集/测试集各类别样本比例与原数据一致
)

class NewData(Dataset):
    # 初始化方法：接收编码后的文本数据和对应的数字标签列表
    def __init__(self, x_encoding, labels):
        self.x_encoding = x_encoding  # 分词器编码后的文本数据
        self.labels = labels  # 对应的数字标签列表

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.x_encoding.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

x_train_encoding = tokenizer(
    x_train,
    padding=True,  # 填充：将同一批次内的文本填充至该批次的最长文本长度
    truncation=True,  # 截断：超过max_length限制的文本自动截断，避免模型输入过长
    max_length=64  # 文本最大长度限制为64个token，超出部分截断，不足部分填充
)

x_test_encoding = tokenizer(
    x_test,
    padding=True,
    truncation=True,
    max_length=64
)

train_dataset = NewData(x_train_encoding, label_train)
test_dataset = NewData(x_test_encoding, label_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 计算分类准确率函数
def flat_accuracy(logits, labels):
    index = np.argmax(logits, axis=1).flatten()
    labels = labels.flatten()
    # 计算预测正确的样本数占总样本数的比例，即该批次数据的分类准确率
    prob = np.sum(index == labels) / len(labels)
    return prob

# ===== 3. 训练函数 =====
def train():
    model.train()
    total_loss = 0  # 累计当前训练轮次（epoch）的总损失值
    iter_num = 0  # 记录当前迭代的批次序号
    total_iter = len(train_loader)  # 当前训练轮次的总迭代批次数量（总样本数/批次大小）

    for batch in train_loader:
        # 清零优化器梯度：避免上一批次的梯度累积影响当前批次的参数更新
        optim.zero_grad()

        # 提取批次中的输入数据，转换为对应计算设备的张量（与模型设备保持一致）
        input_ids = batch['input_ids'].to(device)  # 文本编码后的索引序列
        attention_mask = batch['attention_mask'].to(device)  # 注意力掩码：标记有效token（1）和填充token（0）
        labels = batch['labels'].to(device)  # 该批次样本的真实数字标签

        # 模型前向传播：传入输入数据和真实标签，自动计算损失值和预测分数（logits）
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]  # 提取当前批次的训练损失值（模型优化的核心目标）

        # 反向传播：计算损失值对模型所有可训练参数的梯度
        loss.backward()
        # 梯度裁剪：将模型参数的梯度最大值限制为1.0，防止BERT训练过程中出现梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()

        # 累计当前批次的损失值，用于后续计算平均训练损失
        total_loss += loss.item()
        iter_num += 1  # 迭代批次序号自增
        if iter_num % 100 == 0:
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    # 训练轮次结束后，打印当前轮次的平均训练损失（总损失/总迭代批次）
    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_loss / len(train_loader)))

# ===== 4. 验证函数 =====
def validation():
    model.eval()
    total_loss = 0  # 累计当前验证轮次的总损失值
    iter_num = 0  # 记录当前迭代的批次序号
    total_accuracy = 0  # 累计当前验证轮次的总准确率
    total_iter = len(test_loader)  # 验证的总迭代批次数量

    # 遍历测试集DataLoader，逐批次获取数据进行模型验证
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 模型前向传播：传入输入数据和真实标签，获取验证损失和预测分数
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # 提取当前批次的验证损失和预测分数（logits）
        loss = outputs[0]
        logits = outputs[1]

        # 累计当前批次的验证损失值
        total_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()

        total_accuracy += flat_accuracy(logits, labels)

    # 计算并打印当前验证轮次的平均准确率（总准确率/总迭代批次）
    avg_accuracy = total_accuracy / total_iter
    print("Accuracy: %.4f" % (avg_accuracy))
    print("Average testing loss: %.4f" % (total_loss / total_iter))

# ===== 6. 推理函数 =====
def predict(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'  # 指定返回PyTorch张量格式的数据，无需额外转换
    )
    # 将编码后的张量数据迁移到对应计算设备上，与模型设备保持一致，保证计算兼容
    encoding = {k: v.to(device) for k, v in encoding.items()}
    # 禁用梯度计算：推理阶段无需更新模型参数，提升推理速度、节省内存资源
    with torch.no_grad():
        outputs = model(**encoding)

    # 提取模型输出的预测分数（logits），包含各类别的预测概率相关信息
    logits = outputs.logits
    # 对预测分数取argmax，得到预测类别对应的数字索引，并转换为Python标量
    index = torch.argmax(logits, dim=-1).item()
    # 将数字索引反向转换为文本标签（使用训练好的LabelEncoder），得到最终分类结果
    pre_label = lbl.inverse_transform([index])[0]

    return pre_label

def test_typical_cases(test_cases):
    """
    批量测试典型样本，打印推理结果
    """
    print("\n--- 典型样本测试结果 ---")
    if not isinstance(test_cases, list):
        test_cases = [test_cases]
    for text in test_cases:
        pred_label = predict(text)
        print(f"输入文本：{text}")
        print(f"预测标签：{pred_label}")

# ===== 7. 主流程 =====
# 定义模型和标签编码器的保存目录及文件路径
save_dir = "./bert_model"
model_weights_path = os.path.join(save_dir, "bert_classifier.pt")  # 模型权重文件保存路径
lbl_path = os.path.join(save_dir, 'label_encoder.pkl')  # 标签编码器文件保存路径

# 检查是否存在已训练完成的模型权重和标签编码器文件
if os.path.exists(model_weights_path) and os.path.exists(lbl_path):
    try:
        model.load_state_dict(torch.load(model_weights_path))
        model.to(device)

        with open(lbl_path, 'rb') as f:
            lbl = pickle.load(f)
        print("模型加载完成！")
    except Exception as e:
        # 若加载失败（文件损坏、格式不匹配等），打印错误信息并重新进行模型训练
        print(f"模型加载失败：{e}，将重新训练...")
        total_epochs = 4  # 设定重新训练的总轮次
        for epoch in range(total_epochs):
            print(f"------------Epoch: {epoch + 1} ----------------")
            train()  # 调用训练函数进行本轮模型训练
            validation()  # 调用验证函数评估本轮训练效果

        # 确保模型保存目录存在，不存在则自动创建，避免保存文件时出错
        os.makedirs(save_dir, exist_ok=True)
        # 保存重新训练后的模型权重，便于后续直接加载使用
        torch.save(model.state_dict(), model_weights_path)

        # 保存重新训练后的标签编码器，保证后续推理的标签映射规则一致
        with open(lbl_path, 'wb') as f:
            pickle.dump(lbl, f)
        print(f"模型已保存到：{save_dir}")

else:
    # 若不存在已训练的模型文件，进行全新的模型训练
    print("未找到已训练模型，开始训练...")
    total_epochs = 4  # 设定全新训练的总轮次
    for epoch in range(total_epochs):
        print(f"------------Epoch: {epoch + 1} ----------------")
        train()  # 调用训练函数进行本轮模型训练
        validation()  # 调用验证函数评估本轮训练效果

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_weights_path)

    # 保存训练完成后的标签编码器，保证后续推理的标签映射规则与训练一致
    with open(lbl_path, 'wb') as f:
        pickle.dump(lbl, f)
    print(f"模型已保存到：{save_dir}")

# 定义测试文本，调用批量测试函数进行分类推理
# 12个类别全覆盖的典型测试用例（每个类别1条，贴合真实交互场景）
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
test_typical_cases(test_cases)
