# 这个模型是一个多任务联合模型，核心结构可以分为三部分：
# 基础特征提取层（BERT）
# 作用：将输入的文本（数字 ID）转换为包含上下文信息的向量表示。
# 输出：
# bert_output[0]：所有 token 的隐藏状态（包括每个字的上下文特征）。
# bert_output[1]：[CLS] token 的隐藏状态（BERT 默认用这个作为整个句子的特征）。

# 意图识别头（sequence_classification）
# 作用：对整个句子进行分类，判断用户意图（如 “查询天气”“预订酒店”）。
# 流程：
# 输入：BERT 的 [CLS] token 特征（768 维）。
# 处理：先经过 Dropout 防止过拟合，再通过全连接层映射到意图类别数（如 10 个意图就输出 10 维向量）。
# 输出：每个意图类别的得分（后续通过 softmax 转换为概率）。

# 槽位填充头（token_classification）
# 作用：对句子中的每个 token（字 / 词）进行分类，识别关键信息（如 “地点”“时间”）。
# 流程：
# 输入：BERT 输出的每个 token 的隐藏状态（768 维）。
# 处理：每个 token 都经过 Dropout 和全连接层，映射到槽位类别数（如 15 个槽位就输出 15 维向量）。
# 输出：每个 token 在每个槽位类别的得分（后续通过 softmax 转换为概率）。

"""
同时完成意图识别和槽位填充的 BERT 模型，这是语义解析任务的经典 “联合建模” 方案
"""
import torch.nn as nn
from transformers import BertModel
from transformers import BertForTokenClassification


class BertForIntentClassificationAndSlotFilling(nn.Module):
    def __init__(self, config):
        super(BertForIntentClassificationAndSlotFilling, self).__init__()
        self.config = config  # 保存配置参数到实例变量（方便后续使用）

        # 加载预训练的BERT模型（核心特征提取器）
        self.bert = BertModel.from_pretrained(config.bert_dir)
        self.bert_config = self.bert.config  # 获取BERT模型的配置

        # 定义领域识别分类头（序列级分类）
        self.domain_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),     # Dropout层：随机失活部分神经元，防止过拟合
            nn.Linear(config.hidden_size, config.domain_num_labels),  # 全连接层：将BERT输出的隐藏层特征（768维）映射到意图类别数
        )

        # 定义意图识别分类头（序列级分类）
        self.sequence_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),     # Dropout层：随机失活部分神经元，防止过拟合
            nn.Linear(config.hidden_size, config.seq_num_labels),  # 全连接层：将BERT输出的隐藏层特征（768维）映射到意图类别数
        )

        # 定义槽填充分类头（Token级分类）
        self.token_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob_ner),
            nn.Linear(config.hidden_size, config.token_num_labels),   # 全连接层：将每个token的隐藏层特征映射到槽位类别数
        )

    # 前向传播方法，定义模型的计算流程
    def forward(self,
                input_ids,       # 文本转换的数字ID（形状：[batch_size, seq_len]）
                attention_mask,  # 注意力掩码（形状同上，1表示有效token，0表示填充）
                token_type_ids,  # 句子类型掩码（形状同上，区分不同句子，单句时全为0）
                ):

        # BERT模型输出：包含所有token的隐藏状态和句子级别的池化输出
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)  # 调用BERT模型进行特征提取
        # [CLS]Token 的池化输出（句子级特征）
        pooler_output = bert_output[1]  # 768  形状：[batch_size, 768]
        # 所有 Token 的隐藏状态（Token 级特征）
        token_output = bert_output[0]  # seq_len * 768  形状：[batch_size, seq_len, 768]
        # 领域识别：通过序列分类头计算领域预测结果
        domain_output = self.domain_classification(pooler_output)
        # 意图识别：通过序列分类头计算意图预测结果
        seq_output = self.sequence_classification(pooler_output)  # 输出形状：[batch_size, seq_num_labels]（每个意图类别的得分）
        # 槽位填充：通过token分类头计算每个token的槽位预测结果
        token_output = self.token_classification(token_output)  # 输出形状：[batch_size, seq_len, token_num_labels]（每个槽位类别的得分）

        return domain_output, seq_output, token_output  # 返回意图预测和槽位预测结果


