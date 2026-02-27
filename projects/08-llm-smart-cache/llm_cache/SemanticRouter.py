"""
基于语义向量的意图路由模块，核心作用是 "把用户提问归类到预设的业务目标"（如问候、退款），
是大模型应用的 "意图识别 / 业务分流入口"
"""

from typing import Optional, List, Union, Any, Dict, Callable
import redis
import os
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger("SemanticRouter")


class SemanticRouter:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.3
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        self.routes = {}
        self.idx_to_target = {}

        self.index_file = f"{self.name}_routes.index"
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)

                routes_data = self.redis.get(f"{self.name}_routes_config")
                if routes_data:
                    self.routes = json.loads(routes_data)
                    for target in self.routes:
                        if self.routes[target]['embeddings'] is not None:
                            self.routes[target]['embeddings'] = np.array(self.routes[target]['embeddings'])
                    self._rebuild_idx_mapping()
            except Exception as e:
                self.index = None
        else:
            self.index = None

    def _rebuild_idx_mapping(self):
        """重建索引映射"""
        current_idx = 0
        for target, route_data in self.routes.items():
            count = len(route_data['questions'])
            for i in range(count):
                self.idx_to_target[current_idx + i] = target
            current_idx += count

    def add_route(self, questions: List[str], target: str):
        """添加路由规则"""
        start_idx = self.index.ntotal if self.index else 0

        if target not in self.routes:
            self.routes[target] = {'questions': [], 'embeddings': None}

        # 去重添加
        for q in questions:
            if q not in self.routes[target]['questions']:
                self.routes[target]['questions'].append(q)

        embeddings = self.embedding_method(self.routes[target]['questions'])

        # 强制2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        self.routes[target]['embeddings'] = embeddings

        # 初始化或添加索引
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings.astype(np.float32))

        # 保存配置
        self.redis.setex(
            f"{self.name}_routes_config",
            self.ttl,
            json.dumps(self.routes, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        )
        faiss.write_index(self.index, self.index_file)

        # 建立映射
        for i in range(len(questions)):
            self.idx_to_target[start_idx + i] = target

        # 缓存问题
        for q in questions:
            self.redis.setex(f"{self.name}_route_cache:{q}", self.ttl, target)

    def route(self, question: str):
        """匹配路由"""

        # 查精确缓存
        cached_result = self.redis.get(f"{self.name}_route_cache:{question}")
        if cached_result:
            return cached_result.decode()

        # 检查索引
        if self.index is None or self.index.ntotal == 0:
            return None
            
        embedding = self.embedding_method([question])

        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        elif embedding.shape[0] != 1:
            embedding = embedding[0:1, :]

        embedding = embedding.astype(np.float32)

        # 搜索
        k = min(10, self.index.ntotal)
        dis, ind = self.index.search(embedding, k=k)

        idx = ind[0][0]
        distance = dis[0][0]

        # 查映射
        best_target = self.idx_to_target.get(idx)

        if best_target and distance < self.distance_threshold:
            # 缓存结果
            self.redis.setex(f"{self.name}_route_cache:{question}", self.ttl, best_target)
            return best_target

        return None

    def clear_cache(self):
        """清空缓存"""

        self.redis.delete(f"{self.name}_routes_config")
        keys = self.redis.keys(f"{self.name}_route_cache:*")
        if keys:
            self.redis.delete(*keys)

        if os.path.exists(self.index_file):
            os.remove(self.index_file)

        self.index = None
        self.routes = {}
        self.idx_to_target = {}

