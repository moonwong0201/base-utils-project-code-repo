from modelscope.msdatasets import MsDataset  # 从modelscope库导入MsDataset类，用于加载数据集
# from modelscope.datasets.constants import DatasetFormations
import os
import pandas as pd
import json

import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

# ================================================================
# 1) 数据集下载 + 图片保存 + CSV 导出
# ================================================================
MAX_DATA_NUMBER = 1000  # 设定最大处理的数据数量上限为1000条
dataset_id = 'AI-ModelScope/LaTeX_OCR'  # 数据集在ModelScope上的ID
subset_name = 'default'  # 数据集子集名称（默认子集）
split = 'train'  # 指定使用数据集的训练集部分

dataset_dir = 'LaTeX_OCR'  # 本地保存图片的目录名称
csv_path = './latex_ocr_train.csv'  # 保存图片路径和文本描述的CSV文件路径

# 检查目录是否已存在
if not os.path.exists(dataset_dir):
    # 从modelscope下载COCO 2014图像描述数据集
    # ds =  MsDataset.load(dataset_id, subset_name=subset_name, split=split)
    ds = MsDataset.load(
        dataset_id,
        subset_name=subset_name,
        split=split,
        # formation=DatasetFormations.DIRECTORY  # 使用枚举的有效值
    )
    print(len(ds))
    # 设置处理的图片数量上限
    total = min(MAX_DATA_NUMBER, len(ds))  # 确定实际处理的样本数（取上限和实际数量的较小值）

    # 创建保存图片的目录
    os.makedirs(dataset_dir, exist_ok=True)

    # 初始化存储图片路径和描述的列表
    image_paths = []
    texts = []

    for i in range(total):
        # 获取每个样本的信息
        item = ds[i]  # 获取第i个样本的数据
        text = item['text']  # 提取样本中的文本描述（LaTeX公式）
        image = item['image']  # 提取样本中的图片数据（公式图片）

        # 保存图片并记录路径
        image_path = os.path.abspath(f'{dataset_dir}/{i}.jpg')
        image.save(image_path)

        # 将路径和描述添加到列表中
        image_paths.append(image_path)
        texts.append(text)

        # 每处理50张图片打印一次进度
        if (i + 1) % 50 == 0:
            print(f'Processing {i + 1}/{total} images ({(i + 1) / total * 100:.1f}%)')

    # 将图片路径和描述保存为CSV文件，包含两列：图片路径和对应文本
    df = pd.DataFrame({
        'image_path': image_paths,
        'text': texts,
    })

    # 将数据保存为CSV文件
    df.to_csv(csv_path, index=False)  # 将DataFrame保存为CSV文件（不包含索引列）

    print(f'数据处理完成，共处理了{total}张图片')

else:
    print(f'{dataset_dir}目录已存在,跳过数据处理步骤')


# ================================================================
# 2) CSV -> 对话格式 JSON + 训练/验证划分
# ================================================================
csv_path = './latex_ocr_train.csv'
train_json_path = './latex_ocr_train.json'  # 输出的训练集 JSON 文件路径
val_json_path = './latex_ocr_val.json'  # 输出的验证集 JSON 文件路径
df = pd.read_csv(csv_path)

# Create conversation format
conversations = []  # 用于存储所有样本的对话格式数据

# Add image conversations
for i in range(len(df)):
    conversations.append({
        "id": f"identity_{i+1}",
        "conversations": [
            {
                "role": "user",
                "value": f"{df.iloc[i]['image_path']}"  # 用户输入内容：当前样本的图片路径
            },
            {
                "role": "assistant",
                "value": str(df.iloc[i]['text'])  # 助手回复内容：当前样本的 LaTeX 文本
            }
        ]
    })

# print(conversations)
# Save to JSON
# Split into train and validation sets
train_conversations = conversations[:-4]
val_conversations = conversations[-4:]

# Save train set
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_conversations, f, ensure_ascii=False, indent=2)

# Save validation set
with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_conversations, f, ensure_ascii=False, indent=2)


# ================================================================
# 3) 加载模型
# ================================================================
"""
加载 Qwen2.5-VL 多模态模型（支持文本和图像输入）及其相关工具
"""

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(
    "./model/Qwen2.5-VL-7B-Instruct/",
    use_fast=False,
    trust_remote_code=True
)

origin_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "./model/Qwen2.5-VL-7B-Instruct/",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
origin_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
processor = AutoProcessor.from_pretrained("./model/Qwen2.5-VL-7B-Instruct/")


# ================================================================
# 4) 模型训练
# ================================================================
prompt = "你是一个LaText OCR助手,目标是读取用户输入的照片，转换成LaTex公式。"
train_dataset_json_path = "latex_ocr_train.json"
val_dataset_json_path = "latex_ocr_val.json"
output_dir = "./output/Qwen2.5-VL-7B-LatexOCR"
MAX_LENGTH = 8192


def process_func(example):
    """
    将数据集进行预处理，将 “图片路径 + LaTeX 文本” 转换为模型可训练的格式
    """
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    image_file_path = conversation[0]["value"]  # 图片路径
    output_content = conversation[1]["value"]  # LaTeX文本

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_file_path}",
                    "resized_height": 500,
                    "resized_width": 100,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # 1. 处理文本：生成模型输入的对话模板
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本

    # 2. 处理图像：对图片进行预处理（缩放、归一化等）
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）

    # 3. 组合文本和图像，转换为模型输入格式
    inputs = processor(
        text=[text],  # 处理后的文本
        images=image_inputs,  # 处理后的图像
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # 将张量转换为列表
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs

    # 4. 处理目标输出（LaTeX文本）：转换为token ID
    response = tokenizer(f"{output_content}", add_special_tokens=False)

    # 5. 拼接输入和输出，构建训练用的序列
    # input_ids：[指令部分token] + [输出部分token] + [pad_token]
    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    # attention_mask：标记哪些token需要被关注（1=关注，0=不关注）
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    # labels：训练时的损失计算目标（-100表示忽略指令部分，只计算输出部分的损失）
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])  # 图像像素值张量
    # 图像网格信息（调整维度）
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）

    # 返回模型训练所需的所有输入
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


origin_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
train_ds = Dataset.from_json(train_dataset_json_path)
train_dataset = train_ds.map(process_func)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
train_peft_model = get_peft_model(origin_model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 配置Trainer
trainer = Trainer(
    model=train_peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开启模型训练
trainer.train()


# ================================================================
# 5) 模型预测（测试）
# ================================================================
# 模型预测
# ====================测试===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取测试模型，从output_dir中获取最新的checkpoint
load_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}"
print(f"load_model_path: {load_model_path}")
val_peft_model = PeftModel.from_pretrained(origin_model, model_id=load_model_path, config=val_config)

test_image_list = []
for item in train_dataset:
    image_file_path = item["conversations"][0]["value"]
    label = item["conversations"][1]["value"]

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_file_path,
                "resized_height": 100,
                "resized_width": 500,
            },
            {
                "type": "text",
                "text": prompt,
            }
        ]}]

    response = predict(messages, val_peft_model)

    print(f"predict:{response}")
    print(f"gt:{label}\n")
    break
