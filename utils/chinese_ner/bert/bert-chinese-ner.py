import codecs
import os.path
import numpy as np

# 导入transformers相关工具
from transformers import (
    BertTokenizerFast,  # 分词器 BertTokenizer 升级版（rust语言实现，速度更快，支持快速分词）
    BertForTokenClassification,  # BERT用于Token分类的模型（NER任务核心模型，输出每个token的类别概率）
    TrainingArguments,  # 训练参数配置类（封装训练过程中的各类参数，如批次大小、学习率、保存策略等）
    Trainer,  # 训练器类（封装了完整训练流程，无需手动写训练循环，简化开发，支持评估、保存等功能）
    DataCollatorForTokenClassification  # 数据整理器（处理批次数据，统一样本长度，补PAD/截断，适配模型批量输入）
)

from datasets import Dataset  # 自定义数据集（结构化存储数据，支持批量处理、映射等操作，适配Trainer）
import torch  # PyTorch核心库（张量计算、设备管理、梯度求解等深度学习核心操作）
from sklearn.metrics import accuracy_score, classification_report  # 评估模型的工具（计算准确率、F1分数等评估指标）
import warnings  # 警告处理库

# 全局配置：忽略无关警告
warnings.filterwarnings('ignore')

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备：{device}")

MODEL_SAVE_PATH = './ner-bert-model'

# NER任务核心标签类型定义
# 标签说明：
# O：非实体（不属于任何定义的实体类型，占数据集中绝大多数）
# B-ORG：机构名开头（B=Begin，实体的起始位置，标记实体的第一个token）
# I-ORG：机构名中间/结尾（I=Inside，实体的延续位置，标记实体的后续token）
# B-PER：人名开头
# I-PER：人名中间/结尾
# B-LOC：地名开头
# I-LOC：地名中间/结尾
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

# 构建标签与ID的双向映射
# id2label：数字ID → 文本标签
# label2id：文本标签 → 数字ID
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

def load_data():
    """
    数据加载函数：
    功能：从指定路径加载训练集和验证集的文本与对应标签，完成初步数据清洗和格式转换
    返回值：训练文本列表、训练标签数字列表、验证文本列表、验证标签数字列表
    """
    train_texts = codecs.open(
        "msra/train/sentences.txt",
    ).readlines()[:1000]
    train_texts = [text.strip().replace(" ", "") for text in train_texts]
    train_labels = codecs.open(
        "msra/train/tags.txt"
    ).readlines()[:1000]
    train_labels = [label.strip().split(" ") for label in train_labels]
    train_labels = [[label2id[x] for x in label] for label in train_labels]

    test_texts = codecs.open(
        "msra/val/sentences.txt"
    ).readlines()[:100]
    test_texts = [text.strip().replace(" ", "") for text in test_texts]
    test_labels = codecs.open(
        "msra/val/tags.txt"
    ).readlines()[:100]
    test_labels = [label.strip().split(" ") for label in test_labels]
    test_labels = [[label2id[x] for x in label] for label in test_labels]

    return train_texts, train_labels, test_texts, test_labels


def tokenize_and_align_labels(examples, tokenizer):
    """
    分词与标签对齐函数
    功能：对输入的字列表进行分词，同时保证分词后token与原始字标签一一对应，不丢失标注信息
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,  # 截断超长文本，适配max_length
        padding=False,  # 此处不做padding，后续由DataCollator统一处理批次内padding
        max_length=128,  # 文本最大长度，超过部分截断，不足部分后续补PAD（BERT常用128/256/512）
        is_split_into_words=True  # 标记输入为已拆分为字（token）的列表，避免分词器重新拆分
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # 获取每个token对应的原始字索引
        previous_word_idx = None
        label_ids = []
        for ids in word_ids:
            if ids is None:
                # ids为None对应特殊token（[CLS]/[SEP]），标签设为-100，不参与损失计算
                label_ids.append(-100)
            elif ids != previous_word_idx:
                # 对应原始字的第一个token（或非子词），直接取原始标签
                label_ids.append(label[ids])
            else:
                # 对应原始字的子词，标签设为-100，避免重复计算
                label_ids.append(-100)
            previous_word_idx = ids
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels  # 将对齐后的标签加入分词结果，形成模型完整输入
    return tokenized_inputs


def prepare_dataset(texts, tags, tokenizer):
    """
    数据集准备函数：
    功能：将原始文本和标签转换为模型可处理的结构化Dataset格式，批量应用分词与标签对齐
    """
    tokens = [list(text) for text in texts]
    dataset = Dataset.from_dict({
        "tokens": tokens,
        "labels": tags
    })

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,  # 批量处理数据，提升处理速度
        remove_columns=dataset.column_names  # 移除原始未处理字段，只保留模型所需的分词结果
    )
    return tokenized_dataset

def compute_metrics(p):
    """
    评估指标计算函数：
    功能：计算模型训练和验证过程中的各类评估指标，用于监控模型效果，判断模型是否收敛/过拟合
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # 对每个token的logits取argmax，得到预测的类别ID
    # 过滤-100，提取有效预测标签和真实标签
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 展平列表，用于计算整体准确率
    flat_true_predictions = [item for row in true_predictions for item in row]
    flat_true_labels = [item for row in true_labels for item in row]
    accuracy = accuracy_score(flat_true_labels, flat_true_predictions)

    # 生成分类报告，计算每类标签的精确率、召回率、F1分数
    report = classification_report(
        flat_true_labels,
        flat_true_predictions,
        output_dict=True,  # 以字典格式返回结果，便于提取数据
        zero_division=0  # 处理无预测结果的类别，避免除以零错误
    )
    f1_scores = {}
    for label in tag_type:
        if label in report:
            f1_scores[f"{label}_f1"] = report[label]["f1-score"]

    return {
        "accuracy": accuracy,
        **f1_scores
    }


# 模型预测与实体提取
def predict(sentence):
    """
    模型预测函数：
    功能：输入待预测文本，输出提取到的实体（实体内容+实体类型）
    """
    tokens = list(sentence)
    inputs = tokenizer(
        tokens,
        truncation=True,
        padding=True,
        is_split_into_words=True,
        return_tensors='pt',  # 返回PyTorch张量格式，适配模型输入
        max_length=128
    ).to(device)  # 将输入数据移至指定设备
    with torch.no_grad():  # 关闭梯度计算，仅用于推理，提升速度并节省内存
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_labels = [id2label[p.item()] for p in predictions[0]]

    # 标签对齐：剔除特殊token和子词，保留与原始字对应的标签
    word_ids = inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_labels.append(predicted_labels[i])
        previous_word_idx = word_idx

    # 确保标签长度与原始字列表一致，避免索引越界
    if len(aligned_labels) > len(tokens):
        aligned_labels = aligned_labels[:len(tokens)]
    elif len(aligned_labels) < len(tokens):
        aligned_labels.extend(['O'] * (len(tokens) - len(aligned_labels)))

    # 实体提取：拼接连续的B-I标签，形成完整实体
    entities = []
    current_entity = ""  # 存储当前正在拼接的实体内容
    current_type = ""  # 存储当前实体的类型

    for token, label in zip(tokens, aligned_labels):
        if label.startswith('B-'):
            # 遇到新实体的开头，先保存上一个已拼接完成的实体
            if current_entity:
                entities.append((current_entity, current_type))
            # 初始化新实体
            current_entity = token
            current_type = label[2:]  # 提取实体类型（去掉B-前缀）
        elif label.startswith('I-') and current_entity and current_type == label[2:]:
            # 遇到实体的延续部分，拼接至当前实体
            current_entity += token
        else:
            # 遇到非实体标签，保存上一个已拼接完成的实体，重置当前实体
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = ""
            current_type = ""

    # 保存最后一个未被保存的实体（避免句子末尾实体丢失）
    if current_entity:
        entities.append((current_entity, current_type))

    return entities


if __name__ == '__main__':
    train_texts, train_labels, test_texts, test_labels = load_data()

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"模型 '{MODEL_SAVE_PATH}' 已存在，正在加载...")
        model = BertForTokenClassification.from_pretrained(MODEL_SAVE_PATH).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
        print("模型加载完毕！")
    else:
        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-chinese/")
        model = BertForTokenClassification.from_pretrained(
            google-bert/bert-base-chinese/",
            num_labels=len(tag_type),  # 指定分类类别数，适配当前NER任务的标签数量
            id2label=id2label,  # 绑定标签映射，便于模型解读和保存
            label2id=label2id  # 绑定标签映射，保证训练时标签转换的一致性
        ).to(device)

        train_dataset = prepare_dataset(train_texts, train_labels, tokenizer)
        eval_dataset = prepare_dataset(test_texts, test_labels, tokenizer)

        training_args = TrainingArguments(
            output_dir=MODEL_SAVE_PATH,  # 模型和日志的输出目录
            learning_rate=3e-5,  # BERT类模型最优学习率区间（3e-5 ~ 5e-5），过大易震荡，过小收敛慢
            per_device_train_batch_size=16,  # 每个设备的训练批次大小，根据设备内存调整（内存小则调小）
            per_device_eval_batch_size=16,  # 每个设备的验证批次大小，与训练批次保持一致即可
            num_train_epochs=4,  # 训练轮数，4轮足够小数据集收敛，过多易过拟合
            weight_decay=0.01,  # 权重衰减，防止模型过拟合（正则化手段，让模型参数更平缓）
            logging_dir='./logs',  # 训练日志保存目录，可用于TensorBoard可视化训练过程
            logging_steps=100,  # 每训练100步打印一次日志，监控训练进度和指标变化
            save_strategy='epoch',  # 按轮保存模型检查点，每轮训练完成后保存一个快照
            save_total_limit=1,  # 限制保存的检查点数量，只保留最新1个，节省磁盘空间
            metric_for_best_model='eval_loss',  # 以验证集损失作为判断最优模型的指标
            greater_is_better=False,  # 验证集损失越小越好，故设为False（F1分数则设为True）
            eval_strategy="epoch",  # 每轮评估验证集，计算eval_loss，为判断最优模型提供依据（核心配置）
            load_best_model_at_end=True,  # 训练结束后，自动加载最优模型（避免保存最后一轮可能过拟合的模型）
            report_to='none'  # 不将日志上报到第三方平台（如W&B），简化流程，专注本地训练
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True  # 自动补全批次内的短文本，使批次内所有样本长度一致
        )

        trainer = Trainer(
            model=model,  # 待训练的模型实例
            args=training_args,  # 训练参数配置
            train_dataset=train_dataset,  # 训练数据集
            eval_dataset=eval_dataset,  # 验证数据集（用于每轮评估）
            tokenizer=tokenizer,  # 分词器（用于保存模型时同步保存，保证后续加载的一致性）
            data_collator=data_collator,  # 数据整理器（处理批次数据对齐）
            compute_metrics=compute_metrics  # 评估指标函数（计算每轮的准确率、F1分数等）
        )

        print("开始训练...")
        trainer.train()  # 执行模型训练，自动完成训练和每轮评估
        trainer.save_model(MODEL_SAVE_PATH)  # 保存训练后的最优模型到指定目录（核心：解决load_best_model_at_end不自动保存的问题）
        print(f"模型训练完成，并已保存到 '{MODEL_SAVE_PATH}'")

    print("开始评估...")
    eval_dataset = prepare_dataset(test_texts, test_labels, tokenizer)

    eval_data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )

    trainer_for_eval = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        data_collator=eval_data_collator,
        eval_dataset=eval_dataset
    )
    eval_results = trainer_for_eval.evaluate()
    print(f"评估结果: {eval_results}")

    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',  # 人、位置
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
        '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
        '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
        '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故',
        '妈妈让我给舅舅王明送一箱水果到南京市建邺区的万达广场',
        '招商银行股份有限公司上海浦东分行与阿里巴巴集团达成金融科技合作',
        '导演张艺谋的新电影将在西安市曲江新区的大唐不夜城举办首映礼，主演是周冬雨和易烊千玺',
        '我叫李北京，老家在河北省北京市县，和朋友张上海一起在美团科技上班'
    ]

    for sentence in test_sentences:
        try:
            entities = predict(sentence)
            print(f"句子: {sentence}")
            if entities:
                for entity, entity_type in entities:
                    print(f" {entity_type}: {entity}")
            else:
                print(" 未识别到实体\n")
        except Exception as e:
            print(f"处理句子时出错: {sentence}")
            print(f"错误信息: {e}")
            print()
        
