# -*- coding: utf-8 -*-
import json
import pdfplumber
import jieba
from rank_bm25 import BM25Okapi

# 加载原始问题集和汽车知识手册PDF文件
questions = json.load(open("questions.json"))
pdf = pdfplumber.open("汽车知识手册.pdf")

# 解析PDF所有页面，按页存储页码和文本内容，空文本做兜底处理
pdf_content = []
for idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(idx + 1),
        'content': pdf.pages[idx].extract_text() or ''
    })

# 对问题和PDF页面文本分别进行结巴中文分词，为BM25检索做数据准备
question_words = [jieba.lcut(x["question"]) for x in questions]
pdf_words = [jieba.lcut(x["content"]) for x in pdf_content]

# 初始化BM25Okapi检索模型，以分词后的PDF页面为检索语料库
bm25 = BM25Okapi(pdf_words)

# 执行BM25单页检索：为每个问题匹配相似度最高的1个PDF页面，保存Top1结果
for idx in range(len(questions)):
    scores = bm25.get_scores(question_words[idx])  # 计算问题与所有PDF页面的相似度得分
    max_similar_page = scores.argsort()[::-1][0]   # 取相似度最高的页面索引
    questions[idx]['reference'] = 'page_' + str(max_similar_page + 1)  # 赋值对应页码标识

with open("submit_bm25_retrieval_top1.json", "w", encoding="utf-8") as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)

# 执行BM25多页检索：为每个问题匹配相似度最高的10个PDF页面，保存Top10结果
for idx in range(len(questions)):
    scores = bm25.get_scores(question_words[idx])   # 重新计算相似度得分
    max_similar_pages = scores.argsort()[::-1][:10] # 取相似度前10的页面索引
    # 构造页码标识列表赋值给reference字段
    questions[idx]['reference'] = ['page_' + str(i + 1) for i in max_similar_pages]

with open("submit_bm25_retrieval_top10.json", "w", encoding="utf-8") as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
