# -*- coding: utf-8 -*-
# 向量检索，文档分块处理
import json
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer

# 1：文本分块（将长文本按指定字符数拆分）
def split_text_fixed_size(text, chunk_size):
    # 实现：用列表推导式按步长截取文本，返回分块列表
    chunk_texts = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunk_texts


# 2：列表去重（保留元素首次出现的顺序）
def remove_duplicates(input_list):
    # 实现：用集合记录已出现元素，遍历筛选未出现的元素
    seen = set()
    page_list = []
    for page in input_list:
        if page not in seen:
            seen.add(page)
            page_list.append(page)
    return page_list


# 数据读取
questions = json.load(open("questions.json"))
pdf = pdfplumber.open("汽车知识手册.pdf")
pdf_contents = []
for idx in range(len(pdf.pages)):
    text = pdf.pages[idx].extract_text() or ""
    chunk_texts = split_text_fixed_size(text, chunk_size=40)
    for chunk_text in chunk_texts:
        pdf_contents.append({
            'page': 'page_' + str(idx + 1),
            'content': chunk_text
        })

# 向量生成
model = SentenceTransformer("bge_models/BAAI/bge-small-zh-v1.5")
questions_sentences = [x['question'] for x in questions]
pdf_sentences = [x['content'] for x in pdf_contents]

questions_embedding = model.encode(questions_sentences, normalize_embeddings=True, show_progress_bar=True)
pdf_embedding = model.encode(pdf_sentences, normalize_embeddings=True, show_progress_bar=True)

# Top1检索
for idx, feat in enumerate(questions_embedding):
    scores = feat @ pdf_embedding.T
    chunk_idx = scores.argsort()[-1]
    max_similar_page = pdf_contents[chunk_idx]['page']
    questions[idx]['reference'] = max_similar_page

with open("submit_bge_segment_retrieval_top1.json", "w", encoding="utf-8") as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)


# Top10检索
for idx, feat in enumerate(questions_embedding):
    scores = feat @ pdf_embedding.T
    chunk_idx = scores.argsort()[::-1]
    pages = [pdf_contents[i]['page'] for i in chunk_idx]
    questions[idx]['reference'] = remove_duplicates(pages[:100])[:10]

with open("submit_bge_segment_retrieval_top10.json", "w", encoding="utf-8") as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
