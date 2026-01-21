# -*- coding: utf-8 -*-
import json
import pdfplumber
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# ===================== 核心功能：基于文本嵌入的RAG检索 =====================
# 整体逻辑：加载问题和PDF文档数据 → 用两种预训练嵌入模型（BGE/JinaAI）生成文本向量 →
# 计算问题与PDF页面的相似度 → 按相似度排序取Top1/Top10页面 → 保存检索结果（用于后续RAG问答）

# 1. 加载数据：读取问题列表和PDF文档内容
# 读取待回答的问题集
questions = json.load(open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/questions.json"))
# 读取PDF文档，按页面拆分并存储（处理空文本，避免后续报错）
pdf = pdfplumber.open("/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/汽车知识手册.pdf")
pdf_content = []
for i in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(i + 1),
        'content': pdf.pages[i].extract_text() or ''
    })

# 2. 数据预处理：提取问题文本和PDF页面文本（用于嵌入生成）
question_sentences = [x['question'] for x in questions]  # 所有问题的文本列表
pdf_content_sentences = [x['content'] for x in pdf_content]  # 所有PDF页面的文本列表


# 3. 核心检索函数：计算问题与PDF页面的相似度，取TopN页面并保存结果
# 参数说明：
# - question_embedding: 问题的文本嵌入向量
# - pdf_content_embedding: PDF页面的文本嵌入向量
# - top_n: 取相似度最高的N个页面（1或10）
# - save_path: 检索结果保存路径
def retrieve_and_save(question_embedding, pdf_content_embedding, top_n, save_path):
    for idx, embedding in enumerate(question_embedding):
        # 计算单个问题与所有PDF页面的相似度（向量点积）
        scores = embedding @ pdf_content_embedding.T
        # 按相似度降序排序，取TopN页面的索引
        max_index_pages = scores.argsort()[::-1][:top_n]
        # 构造页码标识（page_1/page_2...），赋值给对应问题的reference字段
        if top_n == 1:
            page_num = max_index_pages[0]
            questions[idx]["reference"] = 'page_' + str(page_num + 1)
        else:
            questions[idx]["reference"] = ['page_' + str(i + 1) for i in max_index_pages]
    # 保存检索结果（问题+对应最相关的PDF页码）
    with open(save_path, "w", encoding="utf-8") as up:
        json.dump(questions, up, ensure_ascii=False, indent=4)


# 4. 模型1：使用BGE中文嵌入模型生成向量，执行检索并保存结果
# 加载预训练的BGE-small-zh模型（轻量级中文文本嵌入模型）
model_bge = SentenceTransformer("/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-small-zh-v1.5")
# 生成问题和PDF页面的嵌入向量（归一化，提升相似度计算准确性）
question_embedding_bge = model_bge.encode(question_sentences, normalize_embeddings=True)
pdf_content_embedding_bge = model_bge.encode(pdf_content_sentences, normalize_embeddings=True)
# 分别保存Top1和Top10的检索结果
retrieve_and_save(question_embedding_bge, pdf_content_embedding_bge, 1, "submit_bge_retrieval_top1.json")
retrieve_and_save(question_embedding_bge, pdf_content_embedding_bge, 10, "submit_bge_retrieval_top10.json")


# 5. 模型2：使用JinaAI中文嵌入模型生成向量，执行检索并保存结果
# 加载Jina-embeddings-v2-base-zh模型（另一款高性能中文文本嵌入模型）
model_jinaai = SentenceTransformer("/Users/wangyingyue/materials/大模型学习资料——八斗/models/jinaai/jina-embeddings-v2-base-zh")
# 生成嵌入向量
question_embedding_jina = model_jinaai.encode(question_sentences, normalize_embeddings=True)
pdf_content_embedding_jina = model_jinaai.encode(pdf_content_sentences, normalize_embeddings=True)
# 保存检索结果
retrieve_and_save(question_embedding_jina, pdf_content_embedding_jina, 1, "submit_jina_retrieval_top1.json")
retrieve_and_save(question_embedding_jina, pdf_content_embedding_jina, 10, "submit_jina_retrieval_top10.json")