import yaml  # type: ignore
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import sys
import os
sys.path.append(os.getcwd())  # 把当前工作目录加入 Python 搜索路径，确保能导入同级的 rag_api.py

import numpy as np
import pytest
from rag_api import RAG

"""
针对整个 RAG 系统核心功能的集成测试脚本
它的目的是验证 RAG 流程中的关键组件 ——嵌入模型、重排模型和大语言模型（LLM）是否能够正常工作
"""


# 验证嵌入模型是否能正确加载并生成向量
def test_embeddding_model():
    # 检查配置文件中是否启用了嵌入模型
    if config["rag"]["use_embedding"]:
        # 如果启用，则调用 RAG 实例的 get_embedding 方法
        embedding = RAG().get_embedding("测试文本")
        # 断言返回的结果是一个 numpy 数组
        assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
    else:
        # 如果未启用，则执行一个永远为真的断言，使测试通过
        assert 1 == 1


# 验证重排模型是否能正确加载并对文本对的相关性进行合理评分
def test_rerank_model():
    # 检查配置文件中是否启用了重排模型
    if config["rag"]["use_rerank"]:
        # 定义两对文本，一对语义相似，一对语义相反
        test_pair = [["我今天很开心", "我今天很开心"], ["我今天很开心", "我今天很不开心"]]
        # 调用重排模型对这两对文本进行评分
        embedding = RAG().get_rank(test_pair)
        # 断言返回的结果是一个 numpy 数组
        assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
        # 断言第一对的相关性分数高于第二对
        assert embedding[0] > embedding[1]
    else:
        assert 1 == 1


# 验证大语言模型（LLM）是否能正确加载并根据给定的上下文和提示生成回答
def test_llm():
    # 定义一个聊天消息列表
    messages = [
        {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},
        {"role": "user", "content": "请你作为童话故事大王，写一篇短篇童话故事."}
    ]
    # 调用 chat 方法，生成一个故事
    response = RAG().chat(messages, 0.7, 0.9)
    assert response != None

