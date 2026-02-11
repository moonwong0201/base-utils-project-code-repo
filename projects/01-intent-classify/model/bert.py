import json
import os.path
from typing import Union, List
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification

# 从配置文件导入关键参数：BERT预训练模型路径、微调后权重路径、全量类别列表
from config import DEVICE, BERT_MODEL_PERTRAINED_PATH, BERT_MODEL_PKL_PATH, LABEL_MAP_PATH


def _load_label_map():
    if not os.path.exists(LABEL_MAP_PATH):
        raise Exception(f"标签映射表不存在，请先训练BERT模型！")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    label2id = {k: int(v) for k, v in label_map["label2id"].items()}
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    return label2id, id2label


label2id, id2label = _load_label_map()


def _load_bert_model():
    """加载BERT模型（带异常处理 + 半精度优化）"""
    try:
        # 加载Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)

        # 加载分类模型（动态设置类别数，避免硬编码）
        num_labels = len(label2id)
        model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_PERTRAINED_PATH,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,  # 兼容微调时类别数变化的情况
            label2id=label2id,  # 传入训练时的标签映射
            id2label=id2label
        )

        # 加载微调权重（支持CPU/GPU）
        model.load_state_dict(
            torch.load(BERT_MODEL_PKL_PATH, map_location=DEVICE),
            strict=False  # 兼容权重少量不匹配的情况
        )

        # 模型移到设备 + 半精度优化（减少显存占用，提升速度）
        model = model.to(DEVICE)

        return tokenizer, model

    except FileNotFoundError as e:
        raise Exception(f"模型文件不存在：{e}")
    except Exception as e:
        raise Exception(f"模型加载失败：{str(e)}")


tokenizer, model = _load_bert_model()


class NewsDataset(Dataset):
    """
    自定义PyTorch Dataset：将Tokenizer编码后的文本和标签打包成“模型可读取的样本”
    Dataset是PyTorch中“数据容器”的标准接口，必须实现__getitem__和__len__
    """
    def __init__(self, encodings, labels):
        # 初始化：接收Tokenizer的编码结果（input_ids、attention_mask）和标签
        self.encodings = encodings  # 字典格式：{"input_ids": [...], "attention_mask": [...]}
        self.labels = labels        # 标签列表（测试时用0填充，仅为适配模型输入格式）

    # 读取单个样本
    def __getitem__(self, idx):
        # 从encodings中取出idx对应的input_ids和attention_mask，转为PyTorch张量
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 加入标签（测试时标签无意义，仅因为模型forward需要labels参数，所以填0）
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item  # 返回单个样本：{"input_ids": tensor, "attention_mask": tensor, "labels": tensor}

    # 返回数据集的总样本数（DataLoader需要知道总长度来划分批次）
    def __len__(self):
        return len(self.labels)


def _preprocess_text(text: str) -> str:
    """文本预处理：清洗脏数据，提升BERT语义理解效果"""
    if not isinstance(text, str):
        return ""
    # 1. 转小写 + 去首尾空格
    text = text.strip().lower()
    # 2. 去除特殊符号（保留中文/字母/数字/空格，避免破坏语义）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
    # 3. 去除多余空格
    text = re.sub(r"\s+", " ", text)
    # 4. 空文本兜底
    return text if text else "无内容"


def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    if request_text is None:
        raise ValueError("请求文本不能为空")

    if isinstance(request_text, str):
        request_text = _preprocess_text(request_text)
        request_text = [request_text]
    # 若输入是列表，直接使用
    elif isinstance(request_text, list):
        request_text = [_preprocess_text(t) for t in request_text]
    else:
        raise TypeError(f"不支持的输入格式：{type(request_text)}，仅支持str或list")

    # 用Tokenizer处理文本列表，生成模型需要的输入张量
    test_encoding = tokenizer(
        list(request_text),  # 输入文本列表
        truncation=True,     # 文本长度超过max_length时，截断到max_length
        padding=True,        # 文本长度不足max_length时，用0补全到max_length（保证批次内长度一致）
        max_length=30        # BERT输入的最大文本长度（根据训练时的配置设定，避免过长/过短）
    )

    # 构建Dataset：将编码结果和占位符标签（[0]*len(request_text)）打包
    test_dataset = NewsDataset(test_encoding, [0] * len(request_text))
    # 构建DataLoader：将Dataset转为“批量迭代器”，用于批量推理
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,  # 每次批量处理16个样本
        shuffle=False   # 推理时不需要打乱样本顺序（打乱会导致结果和输入对应不上）
    )

    model.eval()
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            # outputs是一个元组，包含2个元素：(分类损失, 12维logits)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()

        # 计算每个样本的预测类别索引：取logits中最大值对应的索引（12维中最大的那个就是预测类别）
        # np.argmax(logits, axis=1)：按行取最大值的索引（axis=1表示“每行内部比较”）
        # flatten()：将结果展平为1维列表（如[[0], [2]]→[0,2]）
        # pred += ...：将当前批次的预测结果加入总列表
        batch_pred_ids = np.argmax(logits, axis=1).flatten().tolist()
        pred.extend(batch_pred_ids)

    classify_result = [id2label[pred_id] for pred_id in pred]
    return classify_result


if __name__ == "__main__":
    texts = [
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

    result = model_for_bert(texts)
    for i, text in enumerate(texts):
        print(f"文本：{text}")
        print(f"意图：{result[i]}\n")
