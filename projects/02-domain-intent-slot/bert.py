from typing import List, Union
import numpy as np
import torch

from transformers import BertTokenizer
from seqeval.metrics.sequence_labeling import get_entities

from config import Args
from model import BertForIntentClassificationAndSlotFilling

args = Args()


class BertExtract():
    def __init__(self):
        try:
            self.config = args
            self.model = BertForIntentClassificationAndSlotFilling(self.config)
            self.model.load_state_dict(torch.load(self.config.load_dir, map_location=self.config.device))
            self.model.eval()

            self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_dir)

            self.id2seqlabel = self.config.id2seqlabel
            self.id2slot = self.config.id2nerlabel
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None

    def extract(self, texts: Union[str, List[str]]):
        if not self.model:
            raise Exception("BERT模型未加载成功")

        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            results = []
            for text in texts:
                try:
                    encoding = self.tokenizer(
                        text,
                        max_length=self.config.max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]
                    token_type_ids = encoding["token_type_ids"]

                    domain_output, seq_output, token_output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                    # 处理意图预测结果和槽位预测结果
                    domain_output = domain_output.detach().cpu().numpy()
                    seq_output = seq_output.detach().cpu().numpy()
                    token_output = token_output.detach().cpu().numpy()

                    domain_output = np.argmax(domain_output, -1)
                    seq_output = np.argmax(seq_output, -1)
                    token_output = np.argmax(token_output, -1)
                    # print(seq_output, token_output)

                    # 解析结果（提取有效部分）
                    domain_output = domain_output[0]  # 取第一个样本（因为只预测了一条文本）
                    seq_output = seq_output[0]
                    # 截取槽位结果：去除BERT添加的首尾符号，长度与原文本一致
                    token_output = token_output[0]

                    # 计算有效长度（基于attention_mask，而非文本长度）
                    active_len = torch.sum(attention_mask).item()  # 真实token长度（含[CLS]/[SEP]）
                    if active_len > 2:
                        token_pred = token_output[1: active_len - 1]  # 去除[CLS]和[SEP]
                    else:
                        token_pred = []

                    # 将领域ID转为标签文本
                    domain = self.config.id2domlabel[domain_output]
                    # 将槽位ID转为标签文本
                    token_pred = [self.config.id2nerlabel.get(i, "O") for i in token_pred]
                    # 将意图ID转为标签文本
                    intent = self.config.id2seqlabel[seq_output]
                    # get_entities会解析标签序列，提取出实体类型和位置（如B-地点、I-地点组合为一个地点实体）
                    entities = get_entities(token_pred)
                    slots = {}
                    for ent in entities:
                        ent_type, start, end = ent
                        # 防止下标越界
                        if start >= len(text) or end >= len(text):
                            continue
                        slots[ent_type] = text[start: end+1]

                    results.append({
                        "text": text,
                        "domain": domain,
                        "intent": intent,
                        "slots": slots
                    })
                except Exception as e:
                    results.append({
                        "text": text,
                        "domain": None,
                        "intent": None,
                        "slots": [],
                        "error": f"BERT抽取失败：{str(e)}"
                    })
            return results