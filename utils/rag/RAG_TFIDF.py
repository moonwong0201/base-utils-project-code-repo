# -*- coding: utf-8 -*-
"""
项目名称：基于TF-IDF的汽车知识RAG检索
功能描述：将汽车相关问题与PDF手册进行文本匹配，返回Top1/Top10相关PDF页面
开发环境：Python 3.8+
依赖库：见下方import，需提前安装（pip install -r requirements.txt）
GitHub仓库：[你的仓库地址，手动补充]
"""

# ====================== 1. 导入所需依赖库 ======================
import json
import pdfplumber
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # 用于提取TF-IDF特征
from sklearn.preprocessing import normalize

# ====================== 2. 读取数据集（问题JSON + PDF手册） ======================
# TODO: 手动实现：读取questions.json文件
# TODO: 手动实现：打开PDF文件，提取每页的页码和文本内容，存入pdf_content列表
questions = json.load(open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/questions.json"))
pdf = pdfplumber.open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/汽车知识手册.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text() or ''
    })

# ====================== 3. 中文文本分词处理 ======================
# 核心逻辑：jieba分词后用空格连接，适配TFidfVectorizer的输入格式
# TODO: 手动实现：对问题文本和PDF文本分别分词，生成question_words和pdf_content_words
question_words = [' '.join(jieba.lcut(question['question'])) for question in questions]
pdf_content_words = [' '.join(jieba.lcut(x['content'])) for x in pdf_content]

# ====================== 4. 提取TF-IDF特征 ======================
# 核心逻辑：先全局拟合（问题+PDF）学习词汇表和IDF，再分别转换为特征矩阵
# TODO: 手动实现：初始化TFidfVectorizer
# TODO: 手动实现：拟合整个语料库（question_words + pdf_content_words）
# TODO: 手动实现：将问题文本转换为TF-IDF特征矩阵（question_feat）
# TODO: 手动实现：将PDF文本转换为TF-IDF特征矩阵（pdf_content_feat）
tfidf = TfidfVectorizer()
tfidf.fit(question_words + pdf_content_words)

question_feat = tfidf.transform(question_words)
pdf_content_feat = tfidf.transform(pdf_content_words)

# ====================== 5. 特征向量归一化 ======================
# 目的：消除向量长度对相似度的影响，让相似度仅由方向决定
# TODO: 手动实现：对问题和PDF的TF-IDF特征矩阵进行归一化
question_feat = normalize(question_feat)
pdf_content_feat = normalize(pdf_content_feat)

# ====================== 6. 检索匹配：返回Top1相关页面 ======================
# 核心逻辑：向量内积计算相似度 → 稀疏矩阵转密集数组 → 排序找最高分页码
# TODO: 手动实现：遍历每个问题的特征向量
# TODO: 手动实现：计算当前问题与所有PDF页面的相似度（矩阵内积）
# TODO: 手动实现：稀疏矩阵转一维密集数组
# TODO: 手动实现：找到相似度最高的页面索引，转换为页码（page_xxx）
# TODO: 手动实现：将Top1页码存入questions的reference字段
for query_idx, feat in enumerate(question_feat):
    score = feat @ pdf_content_feat.T
    score = score.toarray()[0]
    max_similar_page = score.argsort()[::-1][0]
    questions[query_idx]['reference'] = 'page_' + str(max_similar_page + 1)

# TODO: 手动实现：保存Top1匹配结果到submit_tfidf_retrieval_top1.json
with open('submit_tfidf_retrieval_top1.json', 'w', encoding='utf-8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)

# ====================== 7. 检索匹配：返回Top10相关页面 ======================
# 核心逻辑：与Top1一致，仅取前10个最高分页码
# TODO: 手动实现：重新遍历每个问题的特征向量（或复用之前的score）
# TODO: 手动实现：计算相似度并转换为一维数组
# TODO: 手动实现：找到前10个相似度最高的页面索引，转换为页码列表
# TODO: 手动实现：将Top10页码列表存入questions的reference字段
for query_idx, feat in enumerate(question_feat):
    score = feat @ pdf_content_feat.T
    score = score.toarray()[0]
    max_similar_pages = score.argsort()[::-1][:10]
    questions[query_idx]['reference'] = ['page_' + str(i + 1) for i in max_similar_pages]

# TODO: 手动实现：保存Top10匹配结果到submit_tfidf_retrieval_top10.json
with open('submit_tfidf_retrieval_top10.json', 'w', encoding='utf-8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
