import json
import os.path
import torch
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

from datasets import Dataset

# 模型保存路径
MODEL_SAVE_PATH = "./qa-bert-model"
PRETRAINED_BERT_PATH = "bert-base-chinese"
# 训练集/验证集数据路径
TRAIN_DATA_PATH = "cmrc2018_public/train.json"
DEV_DATA_PATH = "cmrc2018_public/dev.json"

# 最大序列长度
MAX_SEQ_LENGTH = 512
# 滑动窗口步长
STRIDE_LENGTH = 128

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
print(f"当前使用设备: {device}")

def load_data(train_path, dev_path):
    """
    功能：加载训练集和验证集的JSON格式数据
    """
    train_data = json.load(open(train_path))
    dev_data = json.load(open(dev_path))
    return train_data, dev_data


def prepare_dataset(raw_data):
    """
    功能：将原始JSON数据转换为（段落/问题/答案）的结构化列表，适配Hugging Face Dataset格式
    """
    # 初始化三个空列表，用于存储平行数据
    paragraphs = []
    questions = []
    answers = []

    # 遍历原始数据的嵌套结构，提取所需信息
    for paragraph in raw_data["data"]:
        # 提取当前段落的上下文文本
        context = paragraph["paragraphs"][0]["context"]
        # 遍历当前段落下的所有问答对（qas）
        for qa in paragraph["paragraphs"][0]["qas"]:
            paragraphs.append(context)
            questions.append(qa["question"])
            # 构造答案字典，text和answer_start均为列表格式
            answers.append({
                "text": [qa["answers"][0]["text"]],
                "answer_start": [qa["answers"][0]["answer_start"]]
            })

    return paragraphs, questions, answers


def preprocess_function(examples):
    """
    功能：批量预处理数据集，将文本转换为BERT可接受的输入格式（token id、attention mask等），并计算答案的token起止位置
    """
    # 提取批次中的问题和上下文，去除首尾多余空格
    questions = [qa.strip() for qa in examples["question"]]
    contexts = [text.strip() for text in examples["context"]]
    
    # 调用分词器进行批量分词处理，生成模型输入格式
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        stride=STRIDE_LENGTH,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # 提取并删除样本映射，后续用于匹配答案
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    # 提取并删除偏移映射，后续用于计算答案起止位置
    offset_mapping = tokenized_examples.pop('offset_mapping')

    # 初始化答案起止位置列表，用于存储每个样本的训练目标
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    # 遍历每个拆分后的样本，计算对应的答案起止token位置
    for idx, offset in enumerate(offset_mapping):
        # 获取当前拆分样本对应的原始样本索引
        sample_idx = sample_mapping[idx]
        # 获取原始样本的答案信息
        answer = examples['answers'][sample_idx]

        # 处理无答案的情况（当前数据集无此场景，预留容错逻辑）
        if len(answer["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue

        # 提取答案在原始文本中的字符起止位置
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # 获取当前样本的序列类型标识
        sequence_ids = tokenized_examples.sequence_ids(batch_index=idx)

        # 定位上下文在token序列中的起始位置
        i = 0
        while sequence_ids[i] != 1:
            i += 1
        context_start = i
        
        # 定位上下文在token序列中的结束位置
        while sequence_ids[i] == 1 and i < len(sequence_ids):
            i += 1
        context_end = i - 1

        # 判断答案是否超出当前拆分样本的上下文范围
        if offset_mapping[idx][context_start][0] > end_char or offset_mapping[idx][context_end][1] < start_char:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # 定位答案起始位置对应的token索引
            i = context_start
            while i <= context_end and offset_mapping[idx][i][0] < start_char:
                i += 1
            start_position = i

            # 定位答案结束位置对应的token索引
            i = context_end
            while i >= context_start and offset_mapping[idx][i][1] > end_char:
                i -= 1
            end_position = i

            # 保存计算得到的答案起止token位置（作为模型训练的目标值）
            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples


def predict(context, question, model, tokenizer, max_seq_len):
    """
    功能：输入单个上下文和问题，返回模型预测的清洗后答案
    """
    # 将模型移动到指定设备
    model.to(device)
    # 对输入的问题和上下文进行分词处理，生成模型可接受的张量格式
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=MAX_SEQ_LENGTH
    )
    # 将输入张量移动到指定设备，与模型保持一致
    inputs = inputs.to(device)
    
    # 无梯度推理（避免计算梯度，提升推理速度，节省显存）
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取模型输出的起止位置logits（得分越高，对应位置是答案的概率越大）
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # 取logits最大值对应的索引，作为答案的起止token位置
    start_idx = torch.argmax(start_logits, dim=-1).item()
    end_idx = torch.argmax(end_logits, dim=-1).item()

    # 将token id转换为原始token，便于后续拼接答案
    all_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # 截取答案对应的token区间
    answer_tokens = all_tokens[start_idx: end_idx + 1]
    # 将token列表转换为完整文本
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    # 清洗答案文本（去除多余空格和BERT的##子词标记）
    answer = answer.replace(" ", "").replace("##", "")
    return answer


if __name__ == "__main__":
    print("===== 开始初始化与数据加载 =====")

    # 加载原始训练集和验证集JSON数据
    train_raw, dev_raw = load_data(TRAIN_DATA_PATH, DEV_DATA_PATH)

    # 格式化数据，转换为平行列表（段落/问题/答案）
    train_paragraphs, train_questions, train_answers = prepare_dataset(train_raw)
    val_paragraphs, val_questions, val_answers = prepare_dataset(dev_raw)

    # 转换为Hugging Face Dataset格式
    train_dataset_dict = {
        "context": train_paragraphs[:200],
        "question": train_questions[:200],
        "answers": train_answers[:200]
    }
    train_dataset = Dataset.from_dict(train_dataset_dict)
    val_dataset_dict = {
        "context": val_paragraphs[:200],
        "question": val_questions[:200],
        "answers": val_answers[:200]
    }
    val_dataset = Dataset.from_dict(val_dataset_dict)

    # 模型加载逻辑（优先加载已保存的模型，无则初始化预训练模型并训练）
    if os.path.exists(MODEL_SAVE_PATH):
        # 加载已保存的分词器和模型
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
        model = BertForQuestionAnswering.from_pretrained(MODEL_SAVE_PATH)
        model = model.to(device)
    else:
        # 初始化预训练分词器和模型
        print("\n===== 开始加载/初始化模型 =====")
        tokenizer = BertTokenizerFast.from_pretrained(
            "google-bert/bert-base-chinese/")
        model = BertForQuestionAnswering.from_pretrained(
            "google-bert/bert-base-chinese/")
        model = model.to(device)

        print("\n===== 开始预处理数据与模型训练 =====")
        # 对训练集进行批量预处理，去除原始列，保留模型输入列
        tokenized_train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        # 对验证集进行批量预处理
        tokenized_val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        # 配置训练参数（TrainingArguments），控制训练过程
        training_args = TrainingArguments(
            output_dir=MODEL_SAVE_PATH,           # 模型和日志输出目录
            learning_rate=3e-5,                   # 学习率（BERT微调常用3e-5~5e-5）
            per_device_train_batch_size=16,       # 每个设备的训练批次大小
            per_device_eval_batch_size=16,        # 每个设备的验证批次大小
            num_train_epochs=8,                   # 训练轮数
            weight_decay=0.01,                    # 权重衰减（防止过拟合）
            logging_dir='./logs',                 # 日志保存目录
            logging_steps=100,                    # 每100步打印一次日志
            save_strategy='epoch',                # 按轮保存模型
            eval_strategy="epoch",                # 按轮进行验证
            metric_for_best_model='eval_loss',    # 以验证集损失作为最优模型判断标准
            greater_is_better=False,              # 损失值越小越好
            report_to='none',                     # 不使用第三方日志工具（如wandb）
            load_best_model_at_end=True,          # 训练结束后加载最优模型
            no_cuda=(device.type != "cuda"),      # 非CUDA设备禁用CUDA
            save_total_limit=3                    # 最多保存3个模型副本（防止磁盘占满）
        )

        # 初始化数据收集器（默认拼接张量，适配QA任务）
        data_collator = DefaultDataCollator()
        # 初始化Trainer，封装模型、数据和训练参数
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        # 开始模型训练
        print("开始训练QA模型...")
        trainer.train()
        # 训练完成后保存模型和分词器到MODEL_SAVE_PATH
        trainer.save_model()

    # 模型评估流程（使用验证集评估模型性能）
    print("\n===== 开始评估模型 =====")
    # 重新预处理验证集（保证数据格式与训练一致）
    tokenized_val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # 初始化评估专用Trainer
    trainer_for_eval = Trainer(
        model=model,
        eval_dataset=tokenized_val_dataset,
        data_collator=DefaultDataCollator()
    )
    # 执行评估，返回评估结果
    eval_results = trainer_for_eval.evaluate()
    print(f"模型评估结果: {eval_results}")

    # 样本测试可视化
    print("\n===== 开始测试验证集样本 =====")
    test_sample_count = 3
    for i in range(min(test_sample_count, len(val_paragraphs))):
        # 提取单个样本的上下文、问题、预期答案
        context = val_paragraphs[i]
        question = val_questions[i]
        expected_answer = val_answers[i]["text"][0]
        # 调用predict函数获取模型预测答案
        predicted_answer = predict(context, question, model, tokenizer, MAX_SEQ_LENGTH)

        # 打印样本测试结果
        print(f"\n--- 样本 {i + 1} ---")
        print(f"问题: {question}")
        print(f"预期答案: {expected_answer}")
        print(f"预测答案: {predicted_answer}")
        # 判断预期答案是否包含在预测答案中（软匹配，容错索引偏移）
        is_match = expected_answer in predicted_answer
        print(f"匹配结果: {is_match}")
        print()
