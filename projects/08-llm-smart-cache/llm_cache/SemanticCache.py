"""
语义缓存的核心实现，核心逻辑是「用 Faiss 做向量相似度检索 + Redis 存问答对」，
解决 "相似提问直接返回历史回答，不用重复调用 LLM" 的问题。
"""

import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss
from sentence_transformers import SentenceTransformer
import logging  

logger = logging.getLogger("SemanticCache")


class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.2
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        # 加载/初始化Faiss索引
        self.index_file = f"{self.name}.index"
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
            except Exception as e:
                self.index = None
        else:
            self.index = None

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        """存储问答对到缓存"""
        # 统一转为列表
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        try:
            # 1. 获取embedding
            embedding = self.embedding_method(prompt)

            # 强制确保2D
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)

            # 2. 初始化或添加索引
            if self.index is None:
                dim = embedding.shape[1]
                self.index = faiss.IndexFlatL2(dim)

            # 3. 添加到索引
            self.index.add(embedding.astype(np.float32))

            # 4. 保存索引文件
            faiss.write_index(self.index, self.index_file)

            # 5. 存储到Redis
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    pipe.setex(f"{self.name}:key:{q}", self.ttl, a)
                    pipe.rpush(f"{self.name}:list", q)
                    pipe.expire(f"{self.name}:list", self.ttl)

                result = pipe.execute()
                return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: str):
        """查询缓存"""

        if self.index is None or self.index.ntotal == 0:
            return None

        try:
            # 1. 获取embedding
            embedding = self.embedding_method(prompt)

            # 强制确保2D
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            embedding = embedding.astype(np.float32)

            # 2. 搜索
            k = min(100, self.index.ntotal)  # 不能超过索引大小
            dis, ind = self.index.search(embedding, k=k)

            # 3. 检查最相似的结果
            if dis[0][0] > self.distance_threshold:
                return None

            # 4. 获取所有符合条件的索引
            valid_indices = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]

            # 5. 从Redis获取历史问题
            history_prompts = self.redis.lrange(f"{self.name}:list", 0, -1)

            # 安全检查索引范围
            matched_prompts = []
            for idx in valid_indices:
                if idx < len(history_prompts):
                    matched_prompts.append(history_prompts[idx])

            if not matched_prompts:
                return None

            # 6. 获取答案
            keys = [f"{self.name}:key:{q}" for q in matched_prompts]
            answers = self.redis.mget(keys)

            return answers

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def clear_cache(self):
        """清空缓存"""
        try:
            # 删除Redis数据
            history_prompts = self.redis.lrange(f"{self.name}:list", 0, -1)
            if history_prompts:
                self.redis.delete(*[f"{self.name}:key:{q}" for q in history_prompts])
            self.redis.delete(f"{self.name}:list")

            # 删除索引文件
            if os.path.exists(self.index_file):
                os.unlink(self.index_file)

            # 重置内存索引
            self.index = None
            return True

        except Exception as e:
            return False
