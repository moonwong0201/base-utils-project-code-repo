"""
语义缓存的核心实现，核心逻辑是「用 Faiss 做向量相似度检索 + Redis 存问答对」，
解决 “相似提问直接返回历史回答，不用重复调用 LLM” 的问题。
"""

import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss
from sentence_transformers import SentenceTransformer


class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,  # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.2  # 相似度阈值（距离越小越相似）
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
        self.embedding_method = embedding_method  # 保存向量化函数

        # 加载/初始化Faiss索引文件（本地文件存储）
        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")  # 加载已有索引
        else:
            self.index = None  # 无索引文件则初始化为None

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        # 统一转为列表（兼容单/多提问）
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        # 1. 提问文本转向量
        embedding = self.embedding_method(prompt)

        # 2. 初始化Faiss索引（L2距离：欧式距离）
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1])  # embedding.shape[1]是向量维度

        # 3. 向量加入Faiss索引
        self.index.add(embedding)
        # 4. 保存索引到本地文件（每次新增都覆盖）
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:  # 批量操作提升效率
                for q, a in zip(prompt, response):
                    # 5. 存储问答对：Key=命名空间+key+提问文本，Value=回答，带TTL 为了查找答案
                    pipe.setex(f"{self.name}:key:{q}", self.ttl, a)
                    # 6. rpush（尾插），保证List顺序和Faiss索引顺序完全一致 为了找对应关系
                    pipe.rpush(f"{self.name}:list", q)
                    # 刷新List和问答对的TTL，避免缓存过期不一致
                    pipe.expire(f"{self.name}:list", self.ttl)
                    pipe.expire(f"{self.name}:key:{q}", self.ttl)

                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: str):
        # 无Faiss索引 → 直接返回None（没有历史缓存）
        if self.index is None:
            return None

        # 1. 新提问转向量（和存储时用同一个embedding_method，保证向量格式一致）
        embedding = self.embedding_method(prompt)
        # 2. Faiss检索：找最相似的100个向量（dis=距离，ind=索引位置）
        dis, ind = self.index.search(embedding, k=100)
        # 3. 过滤：最相似的向量距离超过阈值 → 无匹配的相似提问，返回None
        if dis[0][0] > self.distance_threshold:
            return None

        # 4. 过滤所有距离小于阈值的索引位置
        filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]
        # 5. 从Redis List中获取所有历史提问（按存入顺序）
        history_prompts = self.redis.lrange(f"{self.name}:list", 0, -1)  # 0到-1表示获取所有元素
        print("history_prompts: ", history_prompts)
        # 6. 根据过滤后的索引，找到对应的历史提问
        filtered_prompts = [history_prompts[i] for i in filtered_ind]

        # 7. 从Redis中获取对应回答
        return self.redis.mget([f"{self.name}:key:{q}" for q in filtered_prompts])

    def clear_cache(self):
        # 1. 获取Redis List中所有历史提问（字节类型）
        history_prompts = self.redis.lrange(f"{self.name}:list", 0, -1)
        # 2. 拼接问答对Key，批量删除
        if history_prompts:
            self.redis.delete(*[f"{self.name}:key:{q}" for q in history_prompts])
        # 3. 删除Redis List（清空历史提问列表）
        self.redis.delete(f"{self.name}:list")
        # 4. 删除本地Faiss索引文件
        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")
        # 5. 重置Faiss索引为None
        self.index = None


if __name__ == "__main__":
    def get_embedding(text):
        model = SentenceTransformer("/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-small-zh-v1.5")

        if isinstance(text, str):
            text = [text]

        embeddings = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )
        embeddings = embeddings.astype(np.float32)
        return embeddings


    embed_cache = SemanticCache(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()  # 先清空历史缓存（避免测试干扰）

    # 批量存储3组问答对
    test_prompts = [
        "怎么退款？",          # 基准提问
        "退款流程是什么？",    # 相似提问（距离≈0.01）
        "如何申请退款？"       # 相似提问（距离≈0.02）
    ]
    test_responses = [
        "退款请联系客服：400-123-4567",
        "退款流程：1.提交申请 2.审核 3.到账",
        "申请退款需提供订单号，联系客服处理"
    ]
    # 批量存储
    store_result = embed_cache.store(prompt=test_prompts, response=test_responses)
    print(f"存储结果：{store_result}")

    # 测试1：查询相似提问
    print("\n测试相似提问")
    query1 = "退款流程？"  # 和“退款流程是什么？”相似
    res1 = embed_cache.call(prompt=query1)
    print(f"查询提问：{query1} → 命中回答：{res1}")

    query2 = "我要申请退款"  # 和“如何申请退款？”相似
    res2 = embed_cache.call(prompt=query2)
    print(f"查询提问：{query2} → 命中回答：{res2}")

    # 查询不相似提问
    print("\n测试不相似提问")
    query3 = "怎么点餐？"  # 和退款无关，距离>0.1
    res3 = embed_cache.call(prompt=query3)
    print(f"查询提问：{query3} → 命中回答：{res3}（None表示未命中）")

    # 清空缓存后查询
    print("\n清空缓存后测试")
    embed_cache.clear_cache()
    query4 = "怎么退款？"
    res4 = embed_cache.call(prompt=query4)
    print(f"清空后查询：{query4} → 命中回答：{res4}（None表示缓存已清空）")