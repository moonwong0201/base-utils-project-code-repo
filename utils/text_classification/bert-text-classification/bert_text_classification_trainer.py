import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
import os
import pickle

# 数据加载：读取数据集并进行预处理
dataset = pd.read_csv(
    "dataset.csv",
    sep='\t',
    header=None
)
# 初始化标签编码器，用于将文本类别标签转换为数字标签
lbl = LabelEncoder()
# 拟合前500条数据的标签，建立文本标签与数字标签的映射关系
lbl.fit(dataset[1].values[:500])
# 按照8:2比例拆分训练集和测试集，stratify保证标签分布均匀
x_train, x_test, labels_train, labels_test = train_test_split(
    list(dataset[0].values[:500]),
    lbl.transform(dataset[1].values[:500]),
    test_size=0.2,
    stratify=dataset[1][:500].values
)

# 加载BERT分词器和分类模型，用于文本编码和后续训练
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=12
)


train_encodings = tokenizer(
    x_train,
    padding=True,
    truncation=True,
    max_length=64
)

test_encodings = tokenizer(
    x_test,
    padding=True,
    truncation=True,
    max_length=64
)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")

# 数据集封装：将编码结果和标签转换为Hugging Face Dataset格式，适配Trainer输入
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': labels_train
})
# 封装测试集数据集，格式与训练集一致
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': labels_test
})

# 定义评估指标计算函数，用于Trainer在评估阶段计算准确率
def compute_metrics(eval_pred):
    # 解包获取模型输出logits和真实标签
    logits, labels = eval_pred
    # 对logits取argmax，获取预测的类别索引
    output = np.argmax(logits, axis=-1)
    # 计算并返回准确率
    return {'accuracy': (output == labels).mean()}

# 配置Trainer训练参数，定义训练过程中的各项设置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_gpu_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)
# 实例化Trainer，整合模型、训练参数、数据集和评估函数
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 定义模型和标签编码器的保存路径，用于断点续训和后续推理
save_dir = "./bert_model_trainer"
model_path = os.path.join(save_dir, "trainer_checkpoint")  # Trainer检查点路径
lbl_path = os.path.join(save_dir, 'label_encoder.pkl')

# 断点续训逻辑：如果已存在训练好的模型和标签编码器，直接加载；否则执行训练
if os.path.exists(os.path.join(model_path, "model.safetensors")) and os.path.exists(lbl_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # 加载标签编码器，保证标签映射关系与训练时一致
    with open(lbl_path, 'rb') as f:
        lbl = pickle.load(f)

    print("模型加载完成！")
else:
    trainer.train()
    trainer.evaluate()
    model = trainer.model
    # 创建保存目录（若不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 保存训练好的模型检查点
    trainer.save_model(model_path)

    # 保存标签编码器，用于后续推理时将数字标签转换为文本标签
    with open(lbl_path, 'wb') as f:
        pickle.dump(lbl, f)

# 定义推理函数，支持单条或多条文本的分类预测
def prediction(texts):
    labels = []
    if not isinstance(texts, list):
        texts = [texts]
    for text in texts:
        encodings = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        encodings = encodings.to(device)

        with torch.no_grad():
            logits = model(**encodings).logits
        index = torch.argmax(logits, dim=-1).cpu().item()
        # 将数字标签转换为文本标签
        label = lbl.inverse_transform([index])[0]
        # 将预测结果添加到列表中
        labels.append(label)
    # 返回所有文本的预测结果
    return labels

# 定义测试用例，覆盖12个不同类别，用于验证模型推理效果
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
# 调用推理函数，获取测试用例的预测结果
pred = prediction(test_cases)

# 遍历打印测试用例和对应的预测结果
for i in range(len(test_cases)):
    print(f"文本：{test_cases[i]}")
    print(f"预测结果：{pred[i]}")
