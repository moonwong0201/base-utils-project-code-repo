"""
负责把文本的向量（Embedding）结果缓存到 Redis 中，避免重复计算。

文本 → MD5哈希 → Redis Key
Embedding向量 → 字节流 → Redis Value（带TTL过期）
查询时：文本→哈希→查Key→字节流→还原向量
"""

import numpy as np
import redis
from typing import Optional, List, Union
import hashlib
import logging  # 日志模块

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EmbeddingsCache")


# 定义嵌入缓存的核心类，所有缓存操作都封装在这个类中
class EmbeddingsCache:
    def __init__(
            self,
            name: str, ttl: int = 3600 * 24,  # name：缓存的命名空间（区分不同业务的缓存）；ttl：缓存过期时间，默认24小时
            redis_url: str = "localhost",     # Redis服务器地址，默认本地
            redis_port: int = 6379,           # Redis端口，默认6379
            redis_password: str = None,       # Redis密码，默认无
    ):
        self.name = name
        self.redis = redis.Redis(  # 创建Redis连接对象，建立与Redis的连接
            host=redis_url,        # Redis地址
            port=redis_port,       # Redis端口
            password=redis_password
        )
        self.ttl = ttl

    # 把 “文本→哈希 key→向量字节流” 存入 Redis，带过期时间
    def store(self, text: Union[List[str], str], embedding: np.ndarray):
        # 参数说明：text可以是单个字符串/字符串列表；embedding是对应的numpy向量数组

        if isinstance(text, str):  # 如果传入的是单个字符串
            text = [text]  # 转为列表，统一后续循环处理逻辑

        try:
            with self.redis.pipeline() as pipe:  # 使用Redis管道（pipeline），批量执行命令，提升效率
                for i, t in enumerate(text):
                    t_code = hashlib.md5(t.encode()).hexdigest()  # 文本转MD5哈希值（32位字符串），作为唯一标识
                    key = f"{self.name}:{t_code}"  # 拼接最终的Redis key：命名空间+哈希值（比如embedding_cache:abc123...）
                    value = embedding[i].astype(np.float32).tobytes()  # 将numpy向量转为字节流（Redis只能存储字符串/字节，不能直接存数组）
                    pipe.setex(key, self.ttl, value)  # 管道添加setex命令：存入key，设置过期时间，值为字节流

                return pipe.execute()  # 执行管道中的所有命令，返回执行结果（成功返回[b'OK']，失败返回异常）
        except redis.RedisError as e:  # 只捕获Redis相关异常
            logger.error(f"Redis存储缓存失败: {e}")
            return -1
        except Exception as e:  # 兜底捕获其他异常（比如数组操作异常）
            logger.error(f"存储缓存未知错误: {e}")
            return -1

    # 删除缓存方法
    def delete(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []  # 初始化要删除的key列表
            for t in text:  # 修正：不需要索引 i # 遍历文本列表（注释里修正了不需要索引i，是合理的）
                t_code = hashlib.md5(t.encode()).hexdigest()  # 文本转哈希值
                key_list.append(f"{self.name}:{t_code}")  # 拼接key并加入列表

            delete_count = self.redis.delete(*key_list)  # 批量删除key，*表示解包列表（比如delete(key1, key2)），返回删除成功的数量
            logger.info(f"成功删除{delete_count}个缓存key: {key_list[:5]}")
            return delete_count
        except Exception as e:
            print(f"Delete error: {e}")
            return -1

    # 查询缓存
    def call(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []  # 初始化要查询的key列表
            for i, t in enumerate(text):
                t_code = hashlib.md5(t.encode()).hexdigest()  # 文本转哈希值
                key_list += [f"{self.name}:{t_code}"]  # 拼接key并加入列表

            results = self.redis.mget(*key_list)  # 批量查询key，返回结果列表（顺序和key_list一致，不存在的key返回None）

            if not results:
                logger.warning("查询缓存结果为空")
                return None

            embeddings = []  # 初始化返回的向量列表
            for result in results:  # 遍历查询结果
                if result is None:  # 如果某个key不存在（缓存未命中）
                    embeddings.append(None)  # 对应位置返回None
                else:  # 缓存命中
                    embedding = np.frombuffer(result, dtype=np.float32)  # 将字节流转回numpy数组（dtype=np.float32是向量的标准类型）
                    embeddings.append(embedding)

            return embeddings  # 返回向量列表（顺序和输入文本列表一致）

        except redis.RedisError as e:
            logger.error(f"Redis查询缓存失败: {e}")
            return None
        except Exception as e:
            logger.error(f"查询缓存未知错误: {e}", exc_info=True)
            return None

    def clear_all(self) -> int:
        """清空当前命名空间下的所有缓存（谨慎使用）"""
        try:
            keys = self.redis.keys(f"{self.name}:*")
            if not keys:
                logger.info("当前命名空间下无缓存可清空")
                return 0
            delete_count = self.redis.delete(*keys)
            logger.info(f"清空{self.name}命名空间下{delete_count}个缓存key")
            return delete_count
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return -1


if __name__ == "__main__":
    embed_cache = EmbeddingsCache(  # 创建缓存实例
        name="embedding_cache",  # 命名空间
        ttl=360,   # 过期时间360秒
        redis_url="localhost",  # Redis本地地址
    )

    def get_embedding(texts):  # 模拟生成向量的函数（实际项目中是调用Embedding模型）
        embeddings = []
        for text in texts:
            embeddings.append(np.random.rand(768).astype(np.float32))  # 生成768维随机向量（常见的Embedding维度）
        return np.array(embeddings)

    texts = ["hello world",
             "明天和我妈去吃冒菜",
             "I love table tennis"]

    print("存储结果：", embed_cache.store(texts, embedding=get_embedding(texts)))
    print("清空前查询 'hello world':", embed_cache.call(text="hello world"))
    print("成功删除的缓存个数：", embed_cache.delete(text="hello world"))

    clear_count = embed_cache.clear_all()
    print(f"清空操作删除的Key数量: {clear_count}")

    print("清空后查询 'hello world':", embed_cache.call(text="hello world"))

    remaining_keys = embed_cache.redis.keys(f"{embed_cache.name}:*")
    print("清空后剩余的Key:", remaining_keys)


