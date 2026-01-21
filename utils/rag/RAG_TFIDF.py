# -*- coding: utf-8 -*-
# 基于TF-IDF的汽车知识RAG检索
import json
import pdfplumber
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # 用于提取TF-IDF特征
from sklearn.preprocessing import normalize

# 读取数据集
questions = json.load(open("questions.json"))
pdf = pdfplumber.open("汽车知识手册.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text() or ''
    })

# 中文文本分词处理
question_words = [' '.join(jieba.lcut(question['question'])) for question in questions]
pdf_content_words = [' '.join(jieba.lcut(x['content'])) for x in pdf_content]

# 提取TF-IDF特征
tfidf = TfidfVectorizer()
tfidf.fit(question_words + pdf_content_words)

question_feat = tfidf.transform(question_words)
pdf_content_feat = tfidf.transform(pdf_content_words)

# 特征向量归一化
question_feat = normalize(question_feat)
pdf_content_feat = normalize(pdf_content_feat)

# 检索匹配：返回Top1相关页面
for query_idx, feat in enumerate(question_feat):
    score = feat @ pdf_content_feat.T
    score = score.toarray()[0]
    max_similar_page = score.argsort()[::-1][0]
    questions[query_idx]['reference'] = 'page_' + str(max_similar_page + 1)

with open('submit_tfidf_retrieval_top1.json', 'w', encoding='utf-8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)

# 检索匹配：返回Top10相关页面
for query_idx, feat in enumerate(question_feat):
    score = feat @ pdf_content_feat.T
    score = score.toarray()[0]
    max_similar_pages = score.argsort()[::-1][:10]
    questions[query_idx]['reference'] = ['page_' + str(i + 1) for i in max_similar_pages]

with open('submit_tfidf_retrieval_top10.json', 'w', encoding='utf-8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
