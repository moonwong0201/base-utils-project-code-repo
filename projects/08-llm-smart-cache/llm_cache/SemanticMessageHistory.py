"""
把用户和大模型的对话历史存在 Redis 里，支持按 “关键词 / 相似度” 检索历史对话，
本质是「对话日志的存储 + 检索工具」
"""

import json

import numpy as np
import redis
from typing import Optional, List, Union, Any, Dict
import Levenshtein  # 计算字符串编辑距离（文字相似度）
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticMessageHistory:
    def __init__(
            self,
            name: str,  # 对话的名字，类似session id
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
    ):
        self.name = name
        self.redis = redis.Redis(  # 创建Redis连接对象，建立和Redis的通信
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.embedding_model = SentenceTransformer("/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-small-zh-v1.5")

    # 获取该会话的全部对话历史
    def get_history(self):
        # 从Redis取key为"semantic_history:{会话名}"的值（对话历史存在这个key里）
        history = self.redis.get(f"semantic_history:{self.name}")
        if not history:  # 如果Redis里没有这个key（首次对话），返回空列表
            return []
        else:  # 有值则把JSON字符串转成Python列表（对话历史是列表格式）
            return json.loads(history)

    # 追加对话消息到 Redis
    def add_message(self, message: Dict[Any, Any]):
        history = self.get_history()  # 先取出已有历史（空列表/已有对话）
        history.extend(message)  # 把新消息追加到历史末尾
        # 把更新后的历史转成JSON字符串，存入Redis并设置过期时间
        self.redis.setex(f"semantic_history:{self.name}", self.ttl, json.dumps(history))

    # 按角色 / 数量取最近的对话历史
    def get_recent(self, role: Optional[Union[str, List[str]]] = None, top_k=10):
        history = self.get_history()  # 先取全部历史
        if role:  # 如果指定了角色（比如"user"），筛选出该角色的消息
            selected_history = [message for message in history if message.get("role", "") == role]
        else:  # 没指定角色，取全部历史
            selected_history = history

        if top_k:  # 如果指定了top_k（比如1），只取最后top_k条（最近的）
            selected_history = selected_history[-top_k:]  # 给大模型传上下文时，只传 “最近 10 轮用户消息”，避免上下文过长

        return selected_history

    # 按关键词 + 文字相似度找相关对话
    def get_relevant(self, content: str, top_k=10):
        history = self.get_history()  # 先取全部历史
        if not history:
            return []

        history_contents = [msg.get("content", "") for msg in history]
        history_embeddings = self.embedding_model.encode(
            history_contents,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        content_embeddings = self.embedding_model.encode(
            content,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).reshape(1, -1)

        similarities = cosine_similarity(content_embeddings, history_embeddings)

        msg_with_sim = list(zip(history, similarities))
        msg_with_sim.sort(key=lambda x: x[1], reverse=True)

        selected_history = [msg for msg, sim in msg_with_sim[: top_k]]

        # # 第一步：筛选出content字段包含输入字符串的消息（比如输入"today"，找含"today"的消息）
        # selected_history = [message for message in history if content in message.get("content", "")]
        # if not selected_history:  # 没有匹配的，返回空列表
        #     return []

        # # 第二步：按「编辑距离相似度」排序（值越高越相似），降序排列
        # selected_history = sorted(
        #     selected_history,
        #     # Levenshtein.ratio(a, b)：计算两个字符串的编辑距离相似度，值在 0~1 之间（1 = 完全相同，0 = 完全不同）
        #     key=lambda message: Levenshtein.ratio(message.get("content", ""), content),
        #     reverse=True
        # )
        # if top_k:  # 取前top_k条最相似的
        #     selected_history = selected_history[:top_k]
        return selected_history

    # 保留最近 top_k 条，删除更早的
    def delete_history(self, top_k=10):
        history = self.get_history()  # 取全部历史
        history = history[-top_k:]    # 只保留最后top_k条
        # 把保留的历史重新存入Redis（覆盖原有值）
        self.redis.setex(f"semantic_history:{self.name}", self.ttl, json.dumps(history))

    # 清空该会话的所有对话历史
    def clear_history(self):
        # 删除Redis里该会话的key，彻底清空历史
        return self.redis.delete(f"semantic_history:{self.name}")


if __name__ == "__main__":
    # 初始化会话历史对象（session名=my-session，Redis本地）
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="localhost",
    )
    history.clear_history()  # 先清空历史（避免测试残留）
    # 追加5条对话消息
    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "what is the weather going to be tomorrow?"},
    ])

    print("get_history", history.get_history())  # 输出全部5条消息
    print("get_recent topk=1", history.get_recent(top_k=1))  # 输出最后1条（user的天气问题）
    print("get_recent role=user", history.get_recent(role="user", top_k=1))  # 输出最近1条user消息（天气问题）

    print("\nget_relevant today", history.get_relevant("It's raining today", top_k=1))  # 输出含"today"且最相似的1条（天气问题）
    print("get_relevant today", history.get_relevant("thanks", top_k=1))   # 输出含"thanks"的1条（llm的回复）

    # history.clear_history()
    # history.add_message([
    #     {"role": "user", "content": "hello, how are you?"},
    #     {"role": "llm", "content": "I'm doing fine, thanks."},
    #     {"role": "user", "content": "what is the weather going to be today?"},
    # ])
    # history.gen()


