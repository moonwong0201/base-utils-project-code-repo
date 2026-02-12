"""
项目的数据预处理模块，核心作用是把原始的 JSON 格式数据（包含文本、意图、槽位），
转换成 BERT 模型能直接接收的输入特征（input_ids、attention_mask 等），
同时完成意图标签和槽位 BIO 标签的数字化转换
"""

import json
import re  # 导入正则表达式库，用于定位槽位在文本中的位置
import torch
from transformers import BertTokenizer
from config import Args


# 封装原始数据
class InputExample:
    def __init__(self, set_type, text, domain_label, seq_label, token_label):
        # 初始化方法：将单条原始数据的关键信息封装成类实例
        self.set_type = set_type  # 数据类型
        self.text = text      # 原始文本
        self.domain_label = domain_label
        self.seq_label = seq_label   # 意图标签
        self.token_label = token_label   # 槽位标签

    def __repr__(self):
        return (f"InputExample(set_type='{self.set_type}', "
                f"text='{self.text}', "
                f"domain_label='{self.domain_label}', "
                f"seq_label='{self.seq_label}', "
                f"token_label={self.token_label})")


# 封装模型输入特征
class InputFeature:
    def __init__(self,
                 input_ids,         # 文本的数字ID（BERT核心输入）
                 attention_mask,    # 注意力掩码（1=有效字符，0=填充字符）
                 token_type_ids,    # 句子类型掩码（单句任务全为0）
                 domain_label_ids,
                 seq_label_ids,     # 意图标签的数字ID（比如QUERY→0）
                 token_label_ids):  # 槽位BIO标签的数字ID（比如B-Src→1）
        # 初始化方法：将模型需要的输入特征和标签封装成类实例
        self.input_ids = input_ids   # 文本转换的数字ID
        self.attention_mask = attention_mask  # 注意力掩码
        self.token_type_ids = token_type_ids  # 句子类型掩码
        self.domain_label_ids = domain_label_ids
        self.seq_label_ids = seq_label_ids    # 意图标签的数字ID
        self.token_label_ids = token_label_ids  # 槽位标签的数字ID

    def __repr__(self):
        return (f"InputExample(input_ids='{self.input_ids}', "
                f"attention_mask='{self.attention_mask}', "
                f"token_type_ids='{self.token_type_ids}', "
                f"domain_label_ids='{self.domain_label_ids}', "
                f"seq_label_ids='{self.seq_label_ids}',"
                f"token_label_ids='{self.token_label_ids}')")


class Processor:
    @classmethod  # 类方法：无需实例化，直接通过类调用
    def get_examples(cls, path, set_type):
        raw_examples = []  # 存储所有InputExample实例的列表
        with open(path, 'r') as fp:
            data = json.load(fp)  # 读取文件内容并解析

        # 遍历每一条原始数据，生成InputExample
        for i, d in enumerate(data):
            text = d['text']  # 从数据中提取原始文本
            domain_label = d['domain']  # 提取领域标签
            seq_label = d['intent']  # 提取意图标签
            token_label = d['slots']  # 提取槽位标签
            # 将当前数据封装成InputExample，添加到列表中
            raw_examples.append(
                InputExample(
                    set_type,  # 数据类型（train/test）
                    text,   # 原始文本
                    domain_label,
                    seq_label,
                    token_label
                )
            )
        return raw_examples  # 返回所有封装好的原始数据实例


def convert_example_to_feature(ex_idx, example, tokenizer, config):
    # 1. 从InputExample中提取数据
    set_type = example.set_type  # 数据类型（train/test）
    text = example.text  # 原始文本
    domain_label = example.domain_label  # 领域标签
    seq_label = example.seq_label  # 意图标签
    token_label = example.token_label  # 槽位标签

    domain_label_ids = config.domlabel2id[domain_label]
    # 2. 处理意图标签：文本标签→数字ID（用config中的映射表）
    seq_label_ids = config.seqlabel2id[seq_label]

    # 3. 处理槽位标签：生成BIO格式的数字ID（核心步骤）
    token_label_ids = [0] * len(text)  # 初始化槽位标签ID列表（0对应"O"标签，非实体）
    for k, v in token_label.items():  # 遍历槽位字典（k=槽位类型，v=槽位值）
        # print(k,v, text)
        # 在 text 中找到所有和 v 完全匹配的子串，并返回一个迭代器，包含每个匹配结果的位置信息
        re_res = re.finditer(v, text)
        for span in re_res:  # 遍历每一个匹配的位置（span是匹配对象）
            entity = span.group()  # 匹配到的实体
            start = span.start()   # 实体在文本中的起始索引
            end = span.end()       # 实体在文本中的结束索引
            # print(entity, start, end)
            # 给起始位置标注"B-槽位类型"
            token_label_ids[start] = config.nerlabel2id['B-' + k]
            # 给非起始位置标注"I-槽位类型"
            for i in range(start + 1, end):
                token_label_ids[i] = config.nerlabel2id['I-' + k]

    # 4. 对齐槽位标签长度（适配BERT的[CLS]和[SEP]符号）
    # BERT会在文本前后加[CLS]（开头）和[SEP]（结尾），所以槽位标签也要对应加2个位置（标签为O，即0）
    if len(token_label_ids) >= config.max_len - 2:  # 若文本过长（减去2个特殊符号后仍超max_len）
        token_label_ids = [0] + token_label_ids + [0]  # 直接加首尾0（后续分词器会截断）
    else:  # 若文本过短，加首尾0后用0填充到max_len
        token_label_ids = [0] + token_label_ids + [0] + [0] * (config.max_len - len(token_label_ids) - 2)
    # print(token_label_ids)

    # 5. 处理文本：按字符拆分（中文BERT通常以字为单位）
    text = [i for i in text]
    # 6. 用BERT分词器将文本转换为模型输入
    # tokenizer.encode(text)：只返回input_ids，没有掩码；
    # tokenizer.encode_plus(text)：返回包含所有输入特征的字典，是 BERT 模型训练 / 推理的标准用法。
    inputs = tokenizer.encode_plus(
        text=text,  # 拆分后的字符列表
        max_length=config.max_len,  # 最大长度（统一样本长度）
        padding='max_length',  # 短文本填充到max_len
        truncation='only_first',  # 长文本截断（只截断第一个句子，这里是单句）
        return_attention_mask=True,   # 返回注意力掩码（1=有效文本，0=填充）默认为True
        return_token_type_ids=True,   # 返回句子类型掩码（单句全为0）默认为True
    )
    # 7. 将分词器输出转换为张量（模型输入必须是张量，且不需要计算梯度）
    input_ids = torch.tensor(inputs['input_ids'], requires_grad=False)   # 文本数字ID张量
    attention_mask = torch.tensor(inputs['attention_mask'], requires_grad=False)  # 注意力掩码张量
    token_type_ids = torch.tensor(inputs['token_type_ids'], requires_grad=False)  # 句子类型掩码张量
    domain_label_ids = torch.tensor(domain_label_ids, requires_grad=False)  # 领域标签ID张量
    seq_label_ids = torch.tensor(seq_label_ids, requires_grad=False)  # 意图标签ID张量
    token_label_ids = torch.tensor(token_label_ids, requires_grad=False)  # 槽位标签ID张量
    # 8. 调试用：打印前3个样本的信息，验证预处理是否正确
    if ex_idx < 3:
        print(f'*** {set_type}_example-{ex_idx} ***')
        print(f'text: {text}')  # 拆分后的字符
        print(f'input_ids: {input_ids}')  # 文本数字ID
        print(f'attention_mask: {attention_mask}')  # 注意力掩码
        print(f'token_type_ids: {token_type_ids}')  # 句子类型掩码
        print(f'domain_label_ids: {domain_label_ids}')  # 领域标签ID
        print(f'seq_label_ids: {seq_label_ids}')    # 意图标签ID
        print(f'token_label_ids: {token_label_ids}')  # 槽位标签ID

    # 9. 将所有特征封装成InputFeature实例
    feature = InputFeature(
        input_ids,
        attention_mask,
        token_type_ids,
        domain_label_ids,
        seq_label_ids,
        token_label_ids,
    )

    return feature  # 返回封装好的模型输入特征


def get_features(raw_examples, tokenizer, args):
    features = []   # 存储所有InputFeature实例的列表
    # 遍历所有InputExample，逐个转换为InputFeature
    for i, example in enumerate(raw_examples):
        feature = convert_example_to_feature(i, example, tokenizer, args)
        features.append(feature)
    return features


if __name__ == '__main__':
    args = Args()
    raw_examples = Processor.get_examples('../data/test_process.json', 'test')
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
    features = get_features(raw_examples, tokenizer, args)
