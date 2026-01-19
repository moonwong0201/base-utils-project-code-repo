import json
import time
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch("http://localhost:9200")

# 1. 生成向量: 加载预训练模型
# 这个模型可以将句子转换为 512 维的向量
print("正在加载 SentenceTransformer 模型...")
model = SentenceTransformer('models/bge_models/BAAI/bge-small-zh-v1.5')
print("模型加载完成。")

test_text = "测试文本"
test_vector = model.encode(test_text)  # 将文本转为向量（返回NumPy数组）
actual_dims = len(test_vector)
print(f"模型实际输出维度: {actual_dims}")

# --- 2. 创建索引和向量字段 ---
# 索引名
index_name = "semantic_search_demo"

# 检查索引是否存在，如果不存在则创建
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"旧索引 '{index_name}' 已删除。")
    time.sleep(2)

print(f"正在创建新索引 '{index_name}'...")
es.indices.create(
    index=index_name,  # 目标索引名
    body={
        "mappings": {
            "properties": {  # properties：具体字段配置
                "text": {"type": "text"},
                # 存储文本对应的向量，核心用于语义检索
                "text_vector": {
                    "type": "dense_vector",  # ES专门存储向量的字段类型
                    "dims": 512,  # 根据模型的输出维度来设置
                    "index": True,  # 允许基于该字段检索（否则只能存储不能搜索）
                    "similarity": "cosine"  # 相似度计算方式：余弦相似度（0~1，越近语义越像）
                }
            }
        }
    }
)
print(f"索引 '{index_name}' 创建成功。")

# --- 3. 插入带有向量的文档 ---
print("\n正在生成并插入文档...")
documents = [
    "人工智能是未来的趋势。",
    "机器学习是人工智能的一个重要分支。",
    "自然语言处理技术让机器理解人类语言。",
    "今天天气真好，适合出去玩。",
    "我最喜欢的运动是篮球和足球。"
]

for doc_text in documents:
    # 生成向量
    # model.encode()返回NumPy数组，需用.tolist()转为Python列表（ES不支持NumPy格式）
    vector = model.encode(doc_text).tolist()

    # 插入文档
    es.index(
        index=index_name,  # 目标索引
        document={
            "text": doc_text,   # 原始文本
            "text_vector": vector   # 文本对应的向量
        }
    )
print("所有文档插入完成。")

# 刷新索引：确保插入的文档立即可被搜索
es.indices.refresh(index=index_name)
time.sleep(1)  # 等待索引刷新

# --- 4. 纯向量检索（语义搜索核心） ---
print("\n--- 执行向量检索 ---")
query_text = "关于AI和未来的技术"

# 将查询文本转换为向量
query_vector = model.encode(query_text).tolist()

# 使用 knn 查询进行向量检索
response = es.search(
    index=index_name,
    body={
        "knn": {  # knn：ES专门的向量检索语法
            "field": "text_vector",
            "query_vector": query_vector,  # 查询向量（与文档向量维度一致）
            "k": 3,  # 返回最相似的前3个文档
            "num_candidates": 10  # 候选文档数：先从所有文档选10个候选，再挑前3
        },
        "fields": ["text"],  # 返回 text 字段
        "_source": False  # 不返回整个文档源
    }
)

print(f"查询文本: '{query_text}'")
print(f"找到 {response['hits']['total']['value']} 个最相关的结果:")

for hit in response['hits']['hits']:
    score = hit['_score']  # 向量相似度得分
    text = hit['fields']['text'][0]
    print(f"得分: {score:.4f}, 内容: {text}")

# 也可以将 knn 查询与其他过滤器结合使用
# 比如，只在包含特定关键词的文档中进行向量搜索
print("\n--- 结合 knn 和 filter 查询 ---")
response_combined = es.search(
    index=index_name,
    body={
        "query": {
            "bool": {  # 用bool组合多个条件（关键词过滤 + 向量检索）
                "must": [
                    {
                        "match": {
                            "text": "技术"
                        }
                    }
                ],
                # 这里的意思是，先从所有文档中挑出和查询向量最相似的前10个，实际只有三个，再进入must中
                "filter": [
                    {
                        "knn": {  # 向量检索条件
                            "field": "text_vector",
                            "query_vector": query_vector,
                            "k": 10,
                            "num_candidates": 10
                        }
                    }
                ]
            }
        },
        "fields": ["text"],
        "_source": False
    }
)

print(f"查询文本: '{query_text}' (并过滤包含 '技术' 的文档)")
print(f"找到 {response_combined['hits']['total']['value']} 个最相关的结果:")

for hit in response_combined['hits']['hits']:
    score = hit['_score']
    text = hit['fields']['text'][0]
    print(f"得分: {score:.4f}, 内容: {text}")
