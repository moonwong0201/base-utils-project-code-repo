import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch
import os
from functools import partial
from tqdm import tqdm

# ===================== 全局配置 & 设备检测 框架 =====================
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# ===================== 数据加载与预处理 框架 =====================
def load_and_preprocess_data(train_data):
    """
    数据预处理框架：适配指令微调格式，转换为HF Dataset
    """
    train_data["input"] = ""
    train_data.columns = ["instruction", "output", "input"]

    ds = Dataset.from_pandas(train_data)
    return ds

# ===================== 模型/分词器初始化 框架 =====================
def initialize_model_and_tokenizer(model_path):
    """
    初始化千问模型与分词器框架（兼容ChatML格式）
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True

    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.to(device)
    return model, tokenizer

# ===================== 训练数据格式化 核心框架 =====================
def process_func(example, tokenizer, label_str, max_length=384):
    """
    单样本格式化框架：转换为ChatML格式，构建训练所需输入/标签
    """
    instruction_text = (f"<|im_start|>system\n现在进行意图分类任务，严格按照以下规则："
                        f"1.只输出标签库中的标签\n"
                        f"2.禁止输出除标签外的任何解释\n"
                        f"3.标签库：{label_str}\n"
                        f"<|im_end|>\n"  # 系统提示：告诉模型任务
                        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"  # 用户输入：instruction+input（input为空）
                        f"<|im_start|>assistant\n")  # 助手前缀：告诉模型“接下来该你输出意图标签了”
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    response = tokenizer(example['output'], add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    labels = [-100] * len(instruction["input_ids"])
    labels += response["input_ids"]
    labels += [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ===================== LoRA低秩微调配置 框架 =====================
def setup_lora(model):
    """
    配置LoRA参数框架，适配千问模型核心层
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ===================== 训练参数配置 框架 =====================
def setup_training_args():
    """
    配置训练参数框架，适配小显存设备
    """
    training_args = TrainingArguments(
        output_dir="./output_Qwen1.5",
        per_device_train_batch_size=6,  # 每个设备的训练批次大小：一次喂给模型6个样本
        gradient_accumulation_steps=4,  # 梯度累积步数：每4个小批次累积一次梯度再更新参数
        # 等效于批次大小=6*4=24
        logging_steps=100,  # 每100步打印一次日志
        do_eval=True,  # 进行评估
        eval_steps=50,  # 每50步评估一次
        num_train_epochs=5,  # 训练5个epoch
        save_steps=50,  # 每50步保存一次模型
        learning_rate=1e-4,
        save_on_each_node=True,  # 在每个节点上保存模型  这里没有什么用处
        gradient_checkpointing=True,  # 启用梯度检查点，节省内存
        report_to="none"
    )
    return training_args


# ===================== 评估指标（准确率） 框架 =====================
def compute_metrics(p):
    """
    计算验证集有效标签预测准确率框架
    """
    logits, labels = p

    predictions = np.argmax(logits, axis=-1)
    valid_mask = (labels != -100)
    valid_prediction = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    accuracy = np.mean(valid_prediction == valid_labels)
    return {
        "eval_accuracy": accuracy
    }


def batch_predict(test_texts, model, tokenizer, label_str):
    pred_result = []

    for text in tqdm(test_texts, desc="预测意图"):
        try:
            result = predict_intent(model, tokenizer, text, label_str)
            pred_result.append(result)
        except Exception as e:
            print(f"预测文本 {text} 时出错：{e}")
            pred_result.append("")

    return pred_result


# ===================== 意图预测（推理） 框架 =====================
def predict_intent(model, tokenizer, text, label_str):
    """
    单文本意图分类推理框架
    """
    messages = [
        {
            "role": "system",
            "content": f"""现在进行意图分类任务，严格按照以下规则：
                          1.只输出标签库中的标签
                          2.禁止输出除标签外的任何解释
                          3.标签库：{label_str}"""
        },
        {"role": "user", "content": text}
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([formatted_text], return_tensors='pt').to(device)

    with torch.no_grad():
        generate_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=5,  # 最多生成新token数量
            do_sample=False,  # 关闭随机采样，生成确定性结果
            temperature=0.0,  # 温度=0：完全确定性（温度越高，输出越随机）
            # top_p=0.95,                           # 核采样：只从概率前95%的token中选（减少无意义输出）
            # repetition_penalty=2.0,               # 重复惩罚：对重复出现的token降低概率（防止模型重复输出）
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,  # 填充token的ID
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = generate_ids[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()

# ===================== 主流程入口 框架 =====================
if __name__ == "__main__":
    """
    主流程骨架：路径配置 → 模型初始化 → 训练/推理分支
    """
    model_path = "Qwen3-0.6B"

    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前运行python脚本文件所在的绝对目录路径
    adapter_path = os.path.join(script_dir, "output_Qwen1.5/")

    base_model, tokenizer = initialize_model_and_tokenizer(model_path)

    train_data = pd.read_csv(
        "intent-dataset.csv",
        sep='\t',
        header=None
    )

    label = set(train_data[1])
    label_str = "、".join(label)

    if os.path.exists(adapter_path) and os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        print(f"发现已存在的LoRA适配器: {adapter_path}")
        print("跳过训练，直接加载模型进行推理...")

        # 加载LoRA适配器到基础模型
        print(f"加载LoRA适配器: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("未发现已存在的LoRA适配器，开始训练...")
        ds = load_and_preprocess_data(train_data)
        split_len = len(ds) * 0.8
        train_ds = Dataset.from_pandas(ds.to_pandas().iloc[: split_len])
        eval_ds = Dataset.from_pandas(ds.to_pandas().iloc[split_len:])

        process_func_with_tokenizer = partial(
            process_func,
            tokenizer=tokenizer,
            label_str=label_str
        )

        train_tokenized = train_ds.map(
            process_func_with_tokenizer,
            remove_columns=train_ds.column_names
        )

        eval_tokenized = eval_ds.map(
            process_func_with_tokenizer,
            remove_columns=eval_ds.column_names
        )

        model = base_model.to(device)

        model.enable_input_require_grads()
        model = setup_lora(model)

        training_args = setup_training_args()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                pad_to_multiple_of=8
            ),
            compute_metrics=compute_metrics
        )

        trainer.train()

        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

    model.eval()

    test_texts = [
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

    results = batch_predict(test_texts, model, tokenizer, label_str)
    for i, text in enumerate(test_texts):
        print(f"输入: {text}")
        print(f"预测意图: {results[i]}")
