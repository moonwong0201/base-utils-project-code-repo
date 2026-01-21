# -*- coding: utf-8 -*-

import json
import pdfplumber
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ---- 加载数据 ----
questions = json.load(open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/questions.json"))
pdf = pdfplumber.open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/汽车知识手册.pdf")
pdf_content = []
for i in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(i + 1),
        'content': pdf.pages[i].extract_text() or ''
    })

# ---- 加载重排序模型 ----
tokenizer = AutoTokenizer.from_pretrained("/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-reranker-base")
rerank_model.eval()

# ---- 召回文件 ----
bge = json.load(open("submit_bge_segment_retrieval_top10.json"))
bm25 = json.load(open("submit_bm25_retrieval_top10.json"))

# ---- 融合 + 重排序 ----
fusion_result = []
k = 60
for q1, q2 in zip(bge, bm25):
    fusion_score = {}
    for idx, page in enumerate(q1['reference']):
        if page not in fusion_score:
            fusion_score[page] = 1 / (idx + k)
        else:
            fusion_score[page] += 1 / (idx + k)

    for idx, page in enumerate(q2['reference']):
        if page not in fusion_score:
            fusion_score[page] = 1 / (idx + k)
        else:
            fusion_score[page] += 1 / (idx + k)

    sorted_dict = sorted(fusion_score.items(), key=lambda x: x[1], reverse=True)
    # 累加 BGE & BM25 得分
    pairs = []
    for pair in sorted_dict[:3]:
        page_index = int(pair[0].split("_")[1]) - 1
        pairs.append([q1['question'], pdf_content[page_index]['content']])

    inputs = tokenizer(
        pairs,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=512
    )
    valid_keys = ["input_ids", "attention_mask", "token_type_ids"]
    inputs = {k: v for k, v in inputs.items() if k in valid_keys}
    with torch.no_grad():
        outputs = rerank_model(**inputs, return_dict=True).logits.view(-1,).float()
    last_index = outputs.argmax().item()
    q1['reference'] = sorted_dict[last_index][0]
    fusion_result.append(q1)

# ---- 输出 ----
with open('submit_fusion_bge+bm25_rerank_retrieval.json', 'w', encoding='utf8') as f:
    json.dump(fusion_result, f, ensure_ascii=False, indent=4)
