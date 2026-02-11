import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
import json

# 数据集路径
DATASET_PATH = '../assets/dataset/dataset.csv'
# 预训练模型路径
BERT_PRETRAINED_PATH = '../assets/models/bert-base-chinese'
# 模型保存路径
BEST_MODEL_WEIGHTS_PATH = '../assets/weights/bert.pt'
# 标签映射表保存路径（关键：推理时加载这个文件）
LABEL_MAP_PATH = '../assets/label2id.json'

# 加载和预处理数据
dataset_df = pd.read_csv(DATASET_PATH, sep='\t', header=None)

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset_df[1].values)
# 提取前500个文本内容
texts = list(dataset_df[0].values)

label2id = {label: idx for idx, label in enumerate(lbl.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

# 创建配置目录（避免路径不存在）
os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
# 保存映射表为JSON（推理时加载，保证顺序一致）
with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump({
        "label2id": label2id,
        "id2label": id2label,
        "classes": list(lbl.classes_)  # 额外保存原始类别列表，方便核对
    }, f, ensure_ascii=False, indent=2)
print(f"标签映射表已保存到：{LABEL_MAP_PATH}")
print(f"标签映射关系：{label2id}")

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)

# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
model = BertForSequenceClassification.from_pretrained(
    BERT_PRETRAINED_PATH,
    num_labels=len(label2id),  # 判别式模型，分多少类
    label2id=label2id,
    id2label=id2label
)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})


# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}


# 配置训练参数
training_args = TrainingArguments(
    output_dir='../assets/weights/bert/',  # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                    # 训练的总轮数
    per_device_train_batch_size=16,        # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,         # 评估时每个设备的批次大小
    warmup_steps=500,                      # 学习率预热的步数，有助于稳定训练
    weight_decay=0.01,                     # 权重衰减，用于防止过拟合
    logging_dir='./logs',                  # 日志存储目录
    logging_steps=100,                     # 每隔100步记录一次日志
    eval_strategy="epoch",                 # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",                 # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,           # 训练结束后加载效果最好的模型
    metric_for_best_model="accuracy",      # 按准确率选最优模型
    greater_is_better=True,                # 准确率越高越好
    seed=42,                               # 固定随机种子，可复现
)

# 实例化 Trainer
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
print("开始评估最优模型...")
eval_results = trainer.evaluate()
print(f"最终测试集准确率：{eval_results['eval_accuracy']:.4f}")

best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    # 加载最优模型（从checkpoint路径加载完整模型）
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    print(f"The best model is located at: {best_model_path}")
    # 保存模型权重（state_dict格式：只保存参数，不保存模型结构，更轻量）
    torch.save(best_model.state_dict(), BEST_MODEL_WEIGHTS_PATH)
    print("Best model saved to assets/weights/bert.pt")
else:
    print("Could not find the best model checkpoint.")
    torch.save(model.state_dict(), BEST_MODEL_WEIGHTS_PATH)
