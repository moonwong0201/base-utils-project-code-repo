# pip install elasticsearch
from elasticsearch import Elasticsearch

ELASTICSEARCH_URL = "http://localhost:9200"

# 如果没有安全认证，直接创建客户端
es_client = Elasticsearch(ELASTICSEARCH_URL)

# 测试连接
if es_client.ping():
    print("连接成功！")
else:
    print("连接失败。请检查 Elasticsearch 服务是否运行。")

# 定义索引名称和映射
index_name = "blog_posts_py"  # 创建一个索引，索引可以理解为数据库表

# 删除索引
if es_client.indices.exists(index=index_name):
    es_client.indices.delete(index=index_name)  # 存在则删除该索引（会清空所有数据）
    print(f"索引 {index_name} 已删除")

# 含有标题、内容、标签、作者、创建时间
# 为创建的 blog_posts_py 索引设定一套完整的数据规则
mapping = {
  "settings": {  # 索引的基本设置 与文档字段无关 决定了索引的存储、备份性能等特性
    "number_of_shards": 1,  # 分片数量（Elasticsearch将数据拆分存储的单元），这里设置为1（不分片）
    "number_of_replicas": 0  # 副本数量（数据的备份），这里设置为0（不备份，适合测试环境）
  },
  "mappings": {  # 字段映射
    "properties": {  # 定义所有字段的属性
      "title": {   # 标题字段
        "type": "text",   # 字段类型：text适合长文本（支持分词和全文搜索）
        "analyzer": "ik_max_word",  # 写入时的分词器：IK最细粒度分词（尽可能拆分多的词语）
        "search_analyzer": "ik_smart"  # 搜索时的分词器：IK智能分词（粗粒度，效率更高）
      },
      "content": {  # 内容字段（配置同title，都是需要全文搜索的文本）
        "type": "text",
        "analyzer": "ik_max_word",
        "search_analyzer": "ik_smart"
      },
      "tags": { "type": "keyword" },  # 标签字段：keyword类型（精确匹配，不分词，适合过滤/聚合）
      "author": { "type": "keyword" },  # 作者字段：keyword类型（适合按作者精确查询）
      "created_at": { "type": "date" }   # 创建时间：date类型（支持日期格式解析和范围查询）
    }
  }
}

# 检查索引是否存在，如果不存在则创建
if not es_client.indices.exists(index=index_name):
    # 调用create方法创建索引，传入索引名和映射配置
    es_client.indices.create(index=index_name, body=mapping)
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已经存在。")


from datetime import datetime

documents = [
    {
      "title": "Elasticsearch 入门指南",
      "content": "这是一篇关于如何安装和使用 Elasticsearch 的详细文章。学习搜索技术可以提升数据处理能力。",
      "tags": ["Elasticsearch", "教程", "搜索"],
      "author": "张三",
      "created_at": datetime(2023, 10, 26, 10, 0, 0)
    },
    {
      "title": "深入理解IK分词器",
      "content": "IK分词器是中文分词的优秀工具。它的智能分词和最细粒度分词模式各有优势。",
      "tags": ["分词", "IK", "中文"],
      "author": "李四",
      "created_at": datetime(2023, 10, 25, 15, 30, 0)
    }
]

for doc in documents:
    es_client.index(index=index_name, document=doc)  # 添加文档操作，相当于数据库的insert
    print(f"文档已插入: '{doc['title']}'")

# 刷新索引，确保文档可被搜索到
es_client.indices.refresh(index=index_name)


# 定义查询函数
def search_docs(query):
    # 调用search()方法执行查询，传入索引名和查询体
    response = es_client.search(index=index_name, body=query)
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    # response['hits']['hits'] 是具体的匹配结果列表
    for hit in response['hits']['hits']:
        # _score 是匹配得分（分数越高，匹配度越高）
        # _source 是文档的原始数据，这里我们打印了标题
        print(f"得分：{hit['_score']}，文档：{hit['_source']['title']}")


# 1. 查询标题中的 "入门指南"
print("\n--- 1. 查询标题中的 '入门指南' ---")
query_1 = {
  "query": {
    "match": {  # match是全文搜索的基本类型，会对查询词分词后匹配
      "title": "入门指南"  # 匹配title字段中包含"入门指南"的文档
    }
  }
}
search_docs(query_1)

# 2. 结合全文和精确匹配查询
print("\n--- 2. 结合全文（搜索技术）和精确匹配（作者：张三） ---")
query_2 = {
  "query": {
    "bool": {   # bool查询用于组合多个条件（与/或/非等关系）
      "must": {  # must：必须满足的条件（影响得分）
        "match": {
          "content": "搜索技术"  # 内容中必须包含"搜索技术"
        }
      },
      "filter": {  # filter：过滤条件（不影响得分，仅用于筛选）
        "term": {  # term是精确匹配（用于keyword类型字段）
          "author": "张三"  # 要求author必须是张三
        }
      }
    }
  }
}
search_docs(query_2)

