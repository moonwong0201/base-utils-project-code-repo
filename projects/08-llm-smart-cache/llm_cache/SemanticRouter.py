"""
基于语义向量的意图路由模块，核心作用是 “把用户提问归类到预设的业务目标”（如问候、退款），
是大模型应用的 “意图识别 / 业务分流入口”
"""

from typing import Optional, List, Union, Any, Dict, Callable
import redis
import os
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticRouter:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.3  # 匹配阈值：距离<0.3才认为匹配成功
    ):
        self.name = name
        self.redis = redis.Redis(  # 创建Redis连接对象：建立与Redis的通信
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        self.routes = {}  # 内存中存储路由配置：{目标名: {'questions': [问题列表], 'embeddings': 向量}}

        # 初始化Faiss向量索引（本地文件持久化）
        self.index_file = f"{self.name}_routes.index"
        if os.path.exists(self.index_file):  # 如果本地有索引文件，加载索引和路由配置（持久化恢复）
            self.index = faiss.read_index(self.index_file)  # 读取本地Faiss索引文件，恢复向量索引
            # 从Redis加载路由配置（JSON字符串→Python字典）
            routes_data = self.redis.get(f"{self.name}_routes_config")
            if routes_data:
                self.routes = json.loads(routes_data)  # JSON字符串转Python字典，恢复路由配置
                # 列表转回 numpy 数组
                for target in self.routes:
                    if self.routes[target]['embeddings'] is not None:
                        self.routes[target]['embeddings'] = np.array(self.routes[target]['embeddings'])

        else:  # 无索引文件，初始化索引为None
            self.index = None

    # 定义添加路由规则的方法：参数是参考问题列表+路由目标
    def add_route(self, questions: List[str], target: str):
        # 如果目标不在路由配置中，初始化该目标的结构
        if target not in self.routes:
            self.routes[target] = {
                'questions': [],  # 该目标下的参考问题列表
                'embeddings': None  # 存储参考问题的向量数组
            }

        # 把新问题添加到路由（去重）
        for q in questions:
            if q not in self.routes[target]['questions']:
                self.routes[target]['questions'].append(q)

        # 为该目标的所有参考问题生成向量
        embeddings = self.embedding_method(self.routes[target]['questions'])
        self.routes[target]['embeddings'] = embeddings

        # 初始化Faiss索引（如果是首次添加路由）
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # L2距离（欧式距离）

        # 把向量添加到Faiss索引
        self.index.add(embeddings)

        # 把路由配置存入Redis（JSON序列化），设置过期时间
        self.redis.setex(
            f"{self.name}_routes_config",
            self.ttl,
            json.dumps(self.routes, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        )
        # 保存Faiss索引到本地文件
        faiss.write_index(self.index, self.index_file)

        # 为每个参考问题创建缓存键（问题→目标，加速匹配）
        for q in questions:
            self.redis.setex(
                f"{self.name}_route_cache:{q}",
                self.ttl,
                target
            )

    # 定义匹配方法：输入用户问题，返回匹配的路由目标（如"早上好"→"greeting"）
    def route(self, question: str):
        # 第一步：查缓存（如果用户问题是参考问题，直接返回目标）
        cached_result = self.redis.get(f"{self.name}_route_cache:{question}")
        if cached_result:
            return cached_result.decode()  # Redis返回字节，转字符串

        # 第二步：生成用户问题的向量（传入列表是为了兼容嵌入函数的输入格式）
        embedding = self.embedding_method([question])

        # 第三步：Faiss搜索最相似的10个参考问题（返回距离+索引）
        # dis：每个匹配结果的L2距离（越小越相似）；ind：每个结果的全局索引
        dis, ind = self.index.search(embedding, k=10)

        # 取最相似的结果（第一个）：全局索引+距离
        idx = ind[0][0]
        distance = dis[0][0]
        print(f"问题：{question} → 匹配到索引{idx}，距离：{distance}")

        # 第四步：根据全局索引找到对应的路由目标
        current_idx = 0  # 累加索引，标记当前目标的索引区间
        best_target = None  # 存储最终匹配的目标

        # 遍历所有路由目标，判断全局索引落在哪个目标的区间内
        for target, route_data in self.routes.items():
            # 当前目标的参考问题数量（索引区间长度）
            count = len(route_data['questions'])

            # 如果全局索引在当前目标的区间内，匹配成功
            if current_idx <= idx < current_idx + count:
                best_target = target  # 记录匹配到的目标
                break
            # 累加索引，进入下一个目标的区间
            current_idx += count

        # 第五步：判断距离是否小于阈值，符合则缓存结果并返回
        if best_target and distance < self.distance_threshold:
            # 把用户问题→目标存入Redis缓存（下次直接返回）
            self.redis.setex(
                f"{self.name}_route_cache:{question}",
                self.ttl,
                best_target
            )
            # 返回匹配的目标
            return best_target
        # 匹配失败，返回None
        return None


if __name__ == "__main__":
    model = SentenceTransformer(
        "/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-small-zh-v1.5")

    # 定义嵌入函数：封装模型的encode方法，返回numpy数组（适配Faiss）
    def embedding_method(texts):
        return model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    # 初始化路由实例：核心参数配置
    router = SemanticRouter(
        name="my_router",
        embedding_method=embedding_method,
        distance_threshold=0.3
    )

    # 1. 删除Redis中的路由配置和缓存
    router.redis.delete(f"{router.name}_routes_config")
    router.redis.delete(f"{router.name}_route_cache:*")  # 删除所有缓存键
    # 2. 删除本地Faiss索引文件
    if os.path.exists(router.index_file):
        os.remove(router.index_file)
    # 3. 重置内存中的索引和路由
    router.index = None
    router.routes = {}

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )

    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    # 测试匹配
    print(router.route("Hi, good morning"))  # 输出：greeting
    print(router.route("早上好"))  # 输出：greeting（语义匹配）
    print(router.route("怎么退货"))  # 输出：refund
    print(router.route("吃饭了吗"))  # 输出：None（匹配失败）
