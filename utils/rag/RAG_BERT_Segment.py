# -*- coding: utf-8 -*-
import json
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------- 工具函数定义 ----------------------
# 函数1：文本分块（将长文本按指定字符数拆分）
def split_text_fixed_size(text, chunk_size):
    # 实现：用列表推导式按步长截取文本，返回分块列表
    chunk_texts = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunk_texts


# 函数2：列表去重（保留元素首次出现的顺序）
def remove_duplicates(input_list):
    # 实现：用集合记录已出现元素，遍历筛选未出现的元素
    seen = set()
    page_list = []
    for page in input_list:
        if page not in seen:
            seen.add(page)
            page_list.append(page)
    return page_list


# ---------------------- 数据读取 ----------------------
# 1. 读取questions.json文件
# 2. 打开汽车知识手册.pdf文件
# 3. 遍历PDF所有页码，提取每页文本并分块，存储页码+分块文本到列表
questions = json.load(open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/questions.json"))
pdf = pdfplumber.open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/汽车知识手册.pdf")
pdf_contents = []
for idx in range(len(pdf.pages)):
    text = pdf.pages[idx].extract_text() or ""
    chunk_texts = split_text_fixed_size(text, chunk_size=40)
    for chunk_text in chunk_texts:
        pdf_contents.append({
            'page': 'page_' + str(idx + 1),
            'content': chunk_text
        })

# ---------------------- 向量生成 ----------------------
# 1. 加载本地BGE-small-zh-v1.5模型
# 2. 提取问题文本列表、PDF分块文本列表
# 3. 生成问题和PDF分块的语义嵌入向量（开启归一化、进度条）
model = SentenceTransformer("/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-small-zh-v1.5")
questions_sentences = [x['question'] for x in questions]
pdf_sentences = [x['content'] for x in pdf_contents]

questions_embedding = model.encode(questions_sentences, normalize_embeddings=True, show_progress_bar=True)
pdf_embedding = model.encode(pdf_sentences, normalize_embeddings=True, show_progress_bar=True)

# ---------------------- Top1检索 ----------------------
# 1. 遍历每个问题的嵌入向量，计算与所有PDF分块的相似度（向量内积）
# 2. 找到相似度最高的分块索引，获取对应页码
# 3. 保存Top1检索结果到JSON文件
for idx, feat in enumerate(questions_embedding):
    scores = feat @ pdf_embedding.T
    chunk_idx = scores.argsort()[-1]
    max_similar_page = pdf_contents[chunk_idx]['page']
    questions[idx]['reference'] = max_similar_page

with open("submit_bge_segment_retrieval_top1.json", "w", encoding="utf-8") as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)


# ---------------------- Top10检索 ----------------------
# 1. 遍历每个问题的嵌入向量，计算相似度并按从高到低排序所有分块
# 2. 提取排序后分块对应的页码（会有重复）
# 3. 对前100个页码去重，取前10个不重复页码
# 4. 保存Top10检索结果到JSON文件
for idx, feat in enumerate(questions_embedding):
    scores = feat @ pdf_embedding.T
    chunk_idx = scores.argsort()[::-1]
    pages = [pdf_contents[i]['page'] for i in chunk_idx]
    questions[idx]['reference'] = remove_duplicates(pages[:100])[:10]

with open("submit_bge_segment_retrieval_top10.json", "w", encoding="utf-8") as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
