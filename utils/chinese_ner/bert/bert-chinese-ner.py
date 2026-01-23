# --------------- 核心模块1：导入依赖与全局配置（基础环境搭建）---------------
# 用于读取文件，支持多种编码格式（比如处理中文文件不容易乱码）
import codecs
import os.path

import numpy as np

# 导入transformers相关工具（BERT模型、分词器、训练器等核心组件）
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

# 全局配置：忽略无关警告（让输出更干净，避免干扰关键信息查看，如版本兼容警告、冗余操作警告）
warnings.filterwarnings('ignore')

# --------------- 全局常量与映射关系定义（整个项目共用的核心配置，一处修改全局生效）---------------
# 步骤1：设备配置（优先使用高性能设备，自动降级兜底，适配不同硬件环境）
# 优先级：MPS（苹果Silicon芯片）> CUDA（NVIDIA显卡）> CPU（兜底，性能最差）
# 说明：MPS/CUDA为硬件加速，能大幅提升训练和预测速度，CPU仅用于无专用加速硬件的场景
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备：{device}")

# 步骤2：模型存储路径（训练完成后模型的保存目录，用于后续加载和预测，无需重新训练）
# 说明：训练完成后，该目录会自动生成模型权重、配置、分词器等核心文件，供后续直接加载
MODEL_SAVE_PATH = './ner-bert-model'

# 步骤3：NER任务核心标签类型定义（所有可能的token标注结果，遵循BIO标注规范）
# 标签说明：
# O：非实体（不属于任何定义的实体类型，占数据集中绝大多数）
# B-ORG：机构名开头（B=Begin，实体的起始位置，标记实体的第一个token）
# I-ORG：机构名中间/结尾（I=Inside，实体的延续位置，标记实体的后续token）
# B-PER：人名开头
# I-PER：人名中间/结尾
# B-LOC：地名开头
# I-LOC：地名中间/结尾
# 补充：BIO规范是NER任务的通用标注方式，能清晰区分实体的边界和内部，便于模型学习
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

# 步骤4：构建标签与ID的双向映射（模型只能处理数字张量，需完成文本标签与数字ID的转换）
# id2label：数字ID → 文本标签（用于模型输出结果的解读，将数字预测结果转换为人类可理解的文本标签）
# label2id：文本标签 → 数字ID（用于原始标注数据的数字化处理，将标注文本转换为模型可训练的数字）
# 说明：双向映射保证了「训练时数字化」和「预测时文本化」的一致性，避免标签错乱
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


# --------------- 核心模块2：数据处理与准备（数据加载、分词、标签对齐，NER任务的核心难点）---------------
def load_data():
    """
    数据加载函数：
    功能：从指定路径加载训练集和验证集的文本与对应标签，完成初步数据清洗和格式转换
    返回值：训练文本列表、训练标签数字列表、验证文本列表、验证标签数字列表
    数据路径说明：
    - 训练文本：./msra/train/sentences.txt（每行一条中文句子，无标注）
    - 训练标签：./msra/train/tags.txt（每行对应一条句子的标注，每个字对应一个标签，用空格分隔）
    - 验证文本：./msra/val/sentences.txt（用于评估模型泛化能力，不参与训练）
    - 验证标签：./msra/val/tags.txt（验证集对应的标注数据）
    处理流程：
    1.  读取文件内容（按行读取，使用codecs避免中文乱码，[:1000]/[:100]为数据切片，加快新手训练速度）
    2.  文本数据清洗（去空格、去换行符，保证文本格式整洁，与标签长度匹配）
    3.  标签数据处理（按空格分割为单个标签列表、通过label2id转换为数字ID，适配模型训练）
    4.  返回处理后的训练集和验证集（格式统一，可直接传入后续数据集准备函数）
    补充：数据切片仅为方便新手快速验证流程，正式项目中应移除切片，使用完整数据集
    """
    train_texts = codecs.open(
        "/Users/wangyingyue/materials/大模型学习资料——八斗/第七周：信息抽取与GraphRAG/Week07/Week07/msra/train/sentences.txt",
    ).readlines()[:1000]
    train_texts = [text.strip().replace(" ", "") for text in train_texts]
    train_labels = codecs.open(
        "/Users/wangyingyue/materials/大模型学习资料——八斗/第七周：信息抽取与GraphRAG/Week07/Week07/msra/train/tags.txt"
    ).readlines()[:1000]
    train_labels = [label.strip().split(" ") for label in train_labels]
    train_labels = [[label2id[x] for x in label] for label in train_labels]

    test_texts = codecs.open(
        "/Users/wangyingyue/materials/大模型学习资料——八斗/第七周：信息抽取与GraphRAG/Week07/Week07/msra/val/sentences.txt"
    ).readlines()[:100]
    test_texts = [text.strip().replace(" ", "") for text in test_texts]
    test_labels = codecs.open(
        "/Users/wangyingyue/materials/大模型学习资料——八斗/第七周：信息抽取与GraphRAG/Week07/Week07/msra/val/tags.txt"
    ).readlines()[:100]
    test_labels = [label.strip().split(" ") for label in test_labels]
    test_labels = [[label2id[x] for x in label] for label in test_labels]

    return train_texts, train_labels, test_texts, test_labels


def tokenize_and_align_labels(examples, tokenizer):
    """
    分词与标签对齐函数（NER任务核心数据处理函数，解决「分词后子词与原始字标签不匹配」的核心问题）
    功能：对输入的字列表进行分词，同时保证分词后token与原始字标签一一对应，不丢失标注信息
    参数说明：
    - examples：输入的数据集字典，必须包含"tokens"（字列表）和"labels"（标签数字列表）两个字段
    - tokenizer：已加载的BertTokenizerFast分词器实例（必须是Fast版本，支持word_ids()获取子词对应原始字索引）
    处理流程：
    1.  对字列表进行分词（生成input_ids、attention_mask等模型输入数据，truncation=True截断超长文本）
    2.  对齐标签（通过word_ids()获取子词对应的原始字索引，处理特殊token和子词）
    3.  特殊token（[CLS]、[SEP]）和子词对应的标签设为-100（模型计算损失时会忽略-100，不参与梯度更新）
    4.  返回包含对齐后标签的分词结果（保证token与标签长度一致，可直接输入模型）
    核心坑点：BERT分词会将部分汉字拆分为子词，直接映射标签会导致错乱，此函数专门解决该问题
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
    参数说明：
    - texts：原始文本列表（如["今天天气很好", "小明去了北京"]）
    - tags：转换后的标签数字列表（如[[0,0,0],[3,4,0,5,6]]）
    - tokenizer：已加载的BertTokenizerFast分词器实例
    处理流程：
    1.  将原始文本拆分为字列表（中文NER按字处理，与训练标注格式一致，保证标签一一对应）
    2.  创建Dataset结构化数据集（封装tokens和labels字段，支持批量映射、筛选等操作）
    3.  批量应用分词与标签对齐函数（batched=True批量处理，提高数据处理效率）
    4.  移除原始字段（tokens、labels），保留分词后的模型输入字段（input_ids、attention_mask、labels）
    5.  返回处理后的结构化数据集（可直接传入Trainer进行训练/评估）
    补充：Dataset格式是transformers库的标准输入格式，与Trainer高度兼容，简化后续流程
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


# --------------- 核心模块3：模型训练与配置（评估指标、训练参数、模型训练，决定模型效果）---------------
def compute_metrics(p):
    """
    评估指标计算函数：
    功能：计算模型训练和验证过程中的各类评估指标，用于监控模型效果，判断模型是否收敛/过拟合
    参数说明：
    - p：模型输出的元组，包含predictions（模型预测结果，logits张量）和labels（真实标签，数字张量）
    处理流程：
    1.  提取预测结果和真实标签，将预测结果从logits转换为类别ID（argmax取概率最大值，得到最终预测类别）
    2.  过滤忽略标签（-100对应的特殊token和子词，不参与评估，保证评估结果的准确性）
    3.  计算整体准确率（accuracy，反映模型整体预测正确率，易被非实体标签O拉高）
    4.  计算分类报告（包含精确率、召回率、F1分数，精准评估每类实体的识别效果，NER任务核心评估指标）
    5.  提取每类标签的F1分数，返回所有评估指标（供Trainer打印和保存，便于后续分析模型优劣）
    返回值：包含准确率和各类标签F1分数的字典（Trainer会自动打印该字典，用于监控训练过程）
    核心说明：F1分数是NER任务的核心评估指标，兼顾精确率（减少误报）和召回率（减少漏报），更能反映模型实用价值
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


# --------------- 核心模块4：模型预测与实体提取（模型加载、预测、结果解析，落地实用功能）---------------
def predict(sentence):
    """
    模型预测函数：
    功能：输入待预测文本，输出提取到的实体（实体内容+实体类型），实现NER任务的落地价值
    参数说明：
    - sentence：待预测的原始文本（如"今天我约了王浩在恭王府吃饭"）
    处理流程：
    1.  将原始文本拆分为字列表（与训练时输入格式保持一致，保证标签对齐逻辑统一）
    2.  分词处理（生成PyTorch张量格式的模型输入，自动补PAD，适配模型输入要求）
    3.  模型预测（关闭梯度计算（torch.no_grad()），提高预测速度，减少内存占用，避免不必要的计算）
    4.  提取预测标签（将模型输出logits转换为文本标签，便于人类理解和后续实体拼接）
    5.  标签对齐（剔除特殊token和子词，保留与原始字列表对应的标签，保证字与标签一一对应）
    6.  长度兜底处理（确保标签长度与原始字列表一致，避免后续实体拼接时索引越界报错）
    7.  实体提取（从对齐后的标签中识别连续实体（B开头+后续I），拼接实体内容，形成完整实体）
    8.  返回提取到的实体列表（格式：[(实体内容, 实体类型), ...]，便于后续打印和处理）
    核心说明：该函数是模型从「训练」到「实用」的桥梁，直接输出业务所需的实体结果
    """
    tokens = list(sentence)
    inputs = tokenizer(
        tokens,
        truncation=True,
        padding=True,
        is_split_into_words=True,
        return_tensors='pt',  # 返回PyTorch张量格式，适配模型输入
        max_length=128
    ).to(device)  # 将输入数据移至指定设备（MPS/CUDA/CPU），与模型设备保持一致，避免报错
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

    # 长度兜底：确保标签长度与原始字列表一致，避免索引越界
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


# --------------- 主程序入口（整合所有模块，执行训练、评估、预测流程，串联整个项目）---------------
if __name__ == '__main__':
    """
    主程序流程（自上而下，串联所有模块，实现端到端的NER任务）：
    1.  加载训练集和验证集数据（调用load_data函数，获取格式化的训练/验证数据）
    2.  判断模型是否已存在：
        - 存在：直接加载预训练模型和分词器（无需重新训练，节省时间）
        - 不存在：加载基础BERT模型，配置训练参数，执行模型训练并保存（完整训练流程）
    3.  加载验证集，对模型进行评估，输出评估结果（监控模型泛化能力，判断模型效果）
    4.  输入测试句子，执行模型预测，输出实体提取结果（验证模型落地效果，直观查看实体提取能力）
    补充：主程序采用「分支判断」设计，实现「一次训练，多次预测」的便捷性
    """
    # 步骤1：加载数据（调用load_data函数，获取处理后的训练集和验证集，为后续训练/评估做准备）
    train_texts, train_labels, test_texts, test_labels = load_data()

    # 步骤2：模型加载/训练分支判断（根据模型目录是否存在，选择不同流程，提高效率）
    if os.path.exists(MODEL_SAVE_PATH):
        """
        分支1：模型已存在，直接加载（快速进入预测流程，无需重复训练）
        处理流程：
        1.  打印加载提示信息（告知用户当前流程，提升交互体验）
        2.  加载预训练的BERT Token分类模型（从指定目录加载，自动适配设备）
        3.  加载对应的分词器（保证分词逻辑与训练时一致，避免预测结果错乱）
        4.  打印加载完成提示（告知用户流程完成，可进入后续评估/预测）
        核心说明：加载的模型包含训练后的权重，具备直接预测的能力，无需重新训练
        """
        print(f"模型 '{MODEL_SAVE_PATH}' 已存在，正在加载...")
        model = BertForTokenClassification.from_pretrained(MODEL_SAVE_PATH).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
        print("模型加载完毕！")
    else:
        """
        分支2：模型不存在，执行训练流程（完整的模型训练流程，生成可用模型）
        处理流程：
        1.  加载基础BERT分词器（使用bert-base-chinese预训练分词器，适配中文文本）
        2.  加载基础BERT Token分类模型，配置类别数和标签映射（初始化模型，适配当前NER任务）
        3.  准备训练集和验证集（调用prepare_dataset函数，转换为结构化Dataset格式）
        4.  配置训练参数（TrainingArguments），设置训练核心配置（批次大小、轮数、保存策略等）
        5.  初始化数据整理器（DataCollatorForTokenClassification），处理批次数据对齐
        6.  初始化Trainer训练器，封装模型、参数、数据集等（整合所有训练要素）
        7.  执行模型训练（调用trainer.train()，自动完成训练和评估流程）
        8.  保存训练后的最优模型（调用trainer.save_model()，将模型保存到指定目录，供后续加载）
        9.  打印训练完成并保存提示（告知用户训练流程完成，模型已就绪）
        核心说明：该分支是模型的「生成流程」，完成后会生成完整的模型目录，后续可直接加载
        """
        tokenizer = BertTokenizerFast.from_pretrained(
            "/Users/wangyingyue/materials/大模型学习资料——八斗/models/google-bert/bert-base-chinese/")
        model = BertForTokenClassification.from_pretrained(
            "/Users/wangyingyue/materials/大模型学习资料——八斗/models/google-bert/bert-base-chinese/",
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

    # 步骤3：模型评估（加载验证集，执行评估，输出评估结果，客观判断模型效果）
    """
    评估流程（独立于训练流程，即使加载已保存模型也能执行，便于后续验证模型效果）：
    1.  打印评估开始提示（告知用户当前流程）
    2.  准备验证数据集（调用prepare_dataset函数，转换为模型可处理的格式）
    3.  初始化评估专用数据整理器（处理批次数据对齐，与训练时一致）
    4.  初始化评估专用Trainer（仅传入模型、评估指标、数据整理器和验证集，无需训练参数）
    5.  执行模型评估，获取评估结果（调用trainer_for_eval.evaluate()，返回各类评估指标）
    6.  打印评估结果（直观查看模型整体准确率和各类实体的F1分数，判断模型优劣）
    核心说明：评估结果是模型效果的客观体现，重点关注各类实体的F1分数，尤其是ORG（机构名）这类难点实体
    """
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

    # 步骤4：测试预测（定义测试句子列表，批量执行预测，输出实体提取结果，直观验证模型落地效果）
    """
    预测流程（面向业务场景，直观展示模型的实体提取能力，验证模型的实用价值）：
    1.  定义测试句子列表（涵盖人名、地名、机构名、易混淆实体等各类场景，全面验证模型能力）
    2.  遍历测试句子，调用predict函数执行预测（逐个处理，避免批量错误）
    3.  异常处理（try-except捕获单个句子的处理错误，避免单个句子失败导致整体流程中断）
    4.  打印每个句子的实体提取结果（格式化输出，便于用户查看和验证，无实体时提示未识别）
    核心说明：测试句子覆盖了日常对话、新闻资讯、易混淆场景，能全面验证模型的鲁棒性和实用性
    """
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