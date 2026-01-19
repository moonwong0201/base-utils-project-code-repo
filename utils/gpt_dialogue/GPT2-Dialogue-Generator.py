# -*- coding: utf-8 -*-
"""
GPT-2 英文对话微调（分离式）
功能：基于GPT2微调对话数据集，支持Beam Search生成回复
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 全局可配置参数
# 设备相关
DEVICE = "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else \
    "cuda" if torch.cuda.is_available() else "cpu"

# 路径相关
MODEL_NAME = "gpt2"  # GPT2模型本地路径
DATA_FILE_PATH = "chat.txt"  # 对话数据集路径
MODEL_SAVE_PATH = "gpt.pt"  # 训练完成后模型权重保存路径

# 训练超参
EPOCHS = 100  # 训练轮数
BATCH_SIZE = 4  # 批次大小
LEARNING_RATE = 8e-5  # 学习率
WEIGHT_DECAY = 0.01  # 权重衰减（防止过拟合）

# 生成超参
GEN_MAX_LEN = 40  # 生成回复的最大长度
BEAM_WIDTH = 4  # Beam Search束宽

# 环境配置（MPS显存优化，针对Mac GPU）
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE_MEMORY_POOL"] = "1"
os.environ["PYTORCH_MPS_FAST_MEMORY"] = "1"

# 1. 加载分词器与预训练模型
# 加载GPT2分词器，设置pad_token（GPT2原生无pad，复用eos_token）
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载GPT2语言模型并移至指定设备（MPS/CUDA/CPU）
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
vocab = tokenizer.get_vocab()  # 获取词汇表字典（新版transformers兼容）


# 2. 自定义对话数据集（加载chat.txt，处理User/AI格式）
class ChatDataset(Dataset):
    """
    自定义Dataset：
    1. 逐行读取chat.txt，筛选User/AI开头的行
    2. 对文本进行tokenize并添加eos标记
    3. 返回(user_input_tensor, ai_answer_tensor)数据对
    """

    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.user_inputs, self.ai_answers = self.load_and_process_data(file_path)

    def load_and_process_data(self, file_path):
        """核心方法：加载并处理数据集，返回张量列表"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到数据集文件，请确认路径正确：{file_path}")

        user_inputs, ai_answers = [], []
        current_user = None  # 临时存储当前User输入，用于匹配后续AI回复

        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            # 处理User输入行
            if line.startswith("User:"):
                user_text = line[6:]  # 截取"User:"后的内容
                # tokenize并添加eos标记（表示输入结束）
                user_token_ids = self.tokenizer.encode(user_text, add_special_tokens=False) + [
                    self.tokenizer.eos_token_id]
                current_user = torch.tensor(user_token_ids, dtype=torch.long)

            # 处理AI回复行（需与前一个User输入匹配）
            elif line.startswith("AI:") and current_user is not None:
                ai_text = line[4:]  # 截取"AI:"后的内容
                ai_token_ids = self.tokenizer.encode(ai_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                current_ai = torch.tensor(ai_token_ids, dtype=torch.long)

                # 存入列表，完成一组User-AI数据匹配
                user_inputs.append(current_user)
                ai_answers.append(current_ai)
                current_user = None  # 重置，准备匹配下一组数据

        # 数据校验（避免格式错误导致后续训练异常）
        if len(user_inputs) != len(ai_answers):
            raise ValueError("数据格式错误：User输入和AI回复数量不匹配")
        if len(user_inputs) == 0:
            raise ValueError("数据集为空：请检查文件内容是否符合User/AI格式")

        return user_inputs, ai_answers

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.user_inputs)

    def __getitem__(self, idx):
        """按索引返回单组样本（User输入，AI回复）"""
        return self.user_inputs[idx], self.ai_answers[idx]

# 3. 数据处理工具（批次对齐、填充）
def pad_sequence(sequences, padding_value=0, length=None):
    """
    序列填充函数：将多个序列补齐到同一长度（方便批次训练）
    :param sequences: 待填充的序列列表
    :param padding_value: 填充值（默认0，此处复用pad_token_id）
    :param length: 目标填充长度（默认取序列中最大值）
    """
    if not sequences:
        return torch.tensor([], dtype=torch.long)

    max_length = max(len(seq) for seq in sequences) if length is None else length
    # 构建填充后的张量（全填充值初始化）
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)

    # 替换填充值为真实序列数据
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]

    return result


def collate_fn(batch):
    """
    DataLoader批量处理函数：对每个批次的样本进行填充对齐
    :param batch: 单个批次的原始数据（列表，元素为__getitem__返回值）
    :return: 填充后的User输入张量、AI回复张量
    """
    if not batch:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    sources, targets = zip(*batch)  # 拆分User输入和AI回复
    max_len = max(max(len(s) for s in sources), max(len(t) for t in targets))  # 取批次内最大长度
    # 对批次内数据进行填充
    sources_padded = pad_sequence(sources, padding_value=tokenizer.pad_token_id, length=max_len)
    targets_padded = pad_sequence(targets, padding_value=tokenizer.pad_token_id, length=max_len)

    return sources_padded, targets_padded


# 4. 训练循环（核心训练逻辑）
def train(model, train_loader, criterion, optimizer, epochs, save_path):
    """
    模型训练函数：
    1. 遍历数据集，计算语言建模损失
    2. 反向传播更新模型参数
    3. 训练完成后保存模型权重
    """
    model.train()  # 模型切换至训练模式（启用Dropout等层）
    for epoch in range(epochs):
        total_loss = 0.0

        # 遍历单个批次的数据
        for user_inputs, ai_answers in train_loader:
            # 数据移至指定设备
            user_inputs = user_inputs.to(DEVICE)
            ai_answers = ai_answers.to(DEVICE)

            # 梯度清零（避免累积）
            optimizer.zero_grad()
            # 模型前向传播，获取预测logits
            outputs = model(user_inputs)
            logits = outputs.logits

            # 调整形状，计算交叉熵损失（忽略pad_token，避免无效损失）
            logits_flat = logits[:, :-1, :].reshape(-1, len(vocab))  # 预测值（去掉最后一个token）
            target_flat = ai_answers[:, 1:].reshape(-1)  # 目标值（去掉第一个token，对齐预测值）
            loss = criterion(logits_flat, target_flat)

            # 反向传播+参数更新
            loss.backward()
            optimizer.step()

            # 累计批次损失
            total_loss += loss.item()

        # 每10轮打印一次平均损失（方便监控训练进度）
        avg_loss = total_loss / len(train_loader)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch + 1:02d}/{epochs:02d}, Average Loss: {avg_loss:.6f}")

    # 保存训练完成后的模型权重
    save_dir = os.path.dirname(save_path)
    # 只有当目录名非空时，才创建目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n模型权重已成功保存至：{save_path}")


# 5. 文本生成（Beam Search策略）
def generate_text_beam_search(model, tokenizer, input_str, max_len, beam_width, device):
    """
    基于Beam Search的文本生成函数：
    1. 输入用户文本，生成连贯的AI回复
    2. 采用束搜索策略，选择最优生成序列
    """
    model.eval()  # 模型切换至评估模式（禁用Dropout等层）
    # 输入文本tokenize并移至设备
    input_tokens = tokenizer.encode(input_str, return_tensors="pt").to(device)
    # 初始化束搜索候选序列（(序列张量, 序列得分)）
    candidates = [(input_tokens, 0.0)]

    with torch.no_grad():  # 禁用梯度计算，节省显存并提升速度
        for _ in range(max_len):
            new_candidates = []

            # 遍历当前所有候选序列
            for candidate, candidate_score in candidates:
                outputs = model(candidate)
                logits = outputs.logits[:, -1, :]  # 取最后一个token的预测结果

                # 选取top-k个最优候选token（k=beam_width）
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)

                # 构建新的候选序列
                for score, next_token in zip(scores.squeeze(0), next_tokens.squeeze(0)):
                    new_candidate = torch.cat((candidate, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                    new_score = candidate_score - score.item()  # 得分取负（方便后续升序排序，值越小越好）

                    # 生成eos_token时终止，否则加入新候选列表
                    if next_token.item() == tokenizer.eos_token_id:
                        new_candidates.append((new_candidate, new_score))
                    else:
                        new_candidates.append((new_candidate, new_score))

            if not new_candidates:
                break
            # 筛选出最优的beam_width个候选序列
            candidates = sorted(new_candidates, key=lambda x: x[1])[:beam_width]

    # 选取得分最优的序列作为生成结果
    best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]
    # 解码张量为文本，截取输入后的生成部分
    input_len = len(tokenizer.encode(input_str))
    output_str = tokenizer.decode(best_candidate.squeeze())[input_len:]

    return output_str


# 6. 主流程
def main():
    """主函数：按顺序执行数据加载、训练、生成测试"""
    # 步骤1：构建数据集和数据加载器
    dataset = ChatDataset(DATA_FILE_PATH, tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练时打乱数据（提升泛化能力）
        collate_fn=collate_fn  # 自定义批次处理函数
    )

    # 步骤2：定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # 忽略pad_token的损失
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 步骤3：判断是否跳过训练（权重文件已存在则加载）
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"检测到已存在模型权重：{MODEL_SAVE_PATH}，直接加载跳过训练...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print("权重加载成功！")
    else:
        print(f"未检测到模型权重，开始训练（共{EPOCHS}轮）...")
        train(model, train_loader, criterion, optimizer, EPOCHS, MODEL_SAVE_PATH)

    # 步骤4：批量测试生成效果
    test_inputs = [
        "what is the weather like today?",
        "hi, how are you?",
        "can you recommend a good book?"
    ]

    print("\n========== 测试生成结果 ==========")
    for i, text in enumerate(test_inputs, start=1):
        generated_text = generate_text_beam_search(
            model, tokenizer, text, GEN_MAX_LEN, BEAM_WIDTH, DEVICE
        )
        print(f"测试 {i}:")
        print(f"User: {text}")
        print(f"AI: {generated_text}\n")


if __name__ == "__main__":
    main()
