from elasticsearch import Elasticsearch
import json

ELASTICSEARCH_URL = "http://localhost:9200"

# 如果没有安全认证，直接创建客户端
es = Elasticsearch(ELASTICSEARCH_URL)

# 检查连接
if es.ping():
    print("成功连接到 Elasticsearch！")
else:
    print("无法连接到 Elasticsearch，请检查服务是否运行。")


def print_search_results(response):
    # response['hits']['total']['value'] 是匹配到的文档总数
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    for hit in response['hits']['hits']:
        # hit['_score']：文档的“匹配得分”（分数越高，与查询条件的相关性越强）
        # hit['_source']：文档的“原始数据”（即插入时的商品信息，如name、price等）
        # json.dumps(...)：格式化打印原始数据，避免中文乱码且缩进美观
        print(f"得分：{hit['_score']}，文档内容：{json.dumps(hit['_source'], ensure_ascii=False, indent=2)}")


# 定义索引名称为products
index_name = "products"

# 删除索引（谨慎操作！会清空所有数据）
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)  # 存在则删除该索引（会清空所有数据）
    print(f"索引 {index_name} 已删除")

# 检查索引是否存在，如果不存在则创建
if not es.indices.exists(index=index_name):
    # 若索引不存在，则创建索引，同时定义“映射（mapping）”（类似数据库的“表结构”）
    es.indices.create(
        index=index_name,   # 指定要创建的索引名
        body={   # 索引的配置内容（包含映射规则）
            "mappings": {  # 映射：定义索引中每个字段的“数据类型”和“处理规则”
                "properties": {  # 定义所有字段的属性
                    "product_id": {"type": "keyword"},  # 商品ID：keyword类型（精确匹配，不分词，适合唯一标识）
                    "name": {"type": "text", "analyzer": "ik_max_word"},   # 商品名称：text类型（支持全文搜索），用IK最细粒度分词
                    "description": {"type": "text", "analyzer": "ik_smart"},  # 商品描述：text类型，用IK智能分词（粗粒度，效率高）
                    "price": {"type": "float"},
                    "category": {"type": "keyword"},  # 商品类别：keyword类型（精确匹配，适合分组/过滤，如“电子产品”）
                    "stock": {"type": "integer"},
                    "on_sale": {"type": "boolean"},  # 是否在售：boolean类型（true/false，适合精确过滤）
                    "created_at": {"type": "date"}
                }
            }
        }
    )
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已存在。")

# 插入一个新文档
doc_1 = {
    "product_id": "A001",
    "name": "智能手机",
    "description": "最新款智能手机，性能强大，拍照清晰。",
    "price": 4999.50,
    "category": "电子产品",
    "stock": 150,
    "on_sale": True,
    "created_at": "2023-01-15T09:00:00Z"
}
# 插入文档到products索引：指定文档ID为“A001”（与product_id一致，便于后续按ID查询/修改）
es.index(index=index_name, id="A001", document=doc_1)
print("文档 'A001' 已插入。")

# 插入另一个文档
doc_2 = {
    "product_id": "B002",
    "name": "无线蓝牙耳机",
    "description": "音质卓越，佩戴舒适，超长续航。",
    "price": 699.00,
    "category": "电子产品",
    "stock": 300,
    "on_sale": True,
    "created_at": "2023-02-20T14:30:00Z"
}
es.index(index=index_name, id="B002", document=doc_2)
print("文档 'B002' 已插入。")

# 刷新索引以确保文档可被搜索到
es.indices.refresh(index=index_name)

# 1. 全文检索（使用 'match' 查询）
# 搜索名称或描述中包含“智能”的商品
print("\n--- 检索 1: 全文检索“智能” ---")
res_1 = es.search(
    index=index_name,
    body={
        "query": {
            "multi_match": {  # multi_match：多字段全文搜索（在多个字段中同时匹配查询词）
                "query": "智能",  # 查询关键词（要搜索的内容）
                "fields": ["name", "description"]  # 表示在商品名称和描述中搜索
            }
        }
    }
)
print_search_results(res_1)


# 2. 结合 'filter' 进行精确过滤
# 搜索价格低于 1000 元且正在促销的电子产品
print("\n--- 检索 2: 结合查询与过滤 ---")
res_2 = es.search(
    index=index_name,
    body={  # 查询条件（ES的DSL查询语法，用Python字典表示）
        "query": {
            "bool": {
                "must": {
                    "match_all": {}  # 匹配所有文档
                },
                # filter：过滤条件（不影响得分，仅筛选符合条件的文档，效率比must高）
                "filter": [  # 这里的条件必须全部满足
                    {"term": {"category": "电子产品"}},  # term：精确匹配（category必须是“电子产品”）
                    {"term": {"on_sale": True}},  # 精确匹配（必须是在售状态）
                    {"range": {"price": {"lt": 1000}}}  # range：范围查询（price < 1000，lt=less than）
                ]
            }
        }
    }
)
print_search_results(res_2)


# 3. 按关键词分组聚合
# 统计不同类别的商品数量
print("\n--- 检索 3: 聚合查询（按类别统计） ---")
res_3 = es.search(
    index=index_name,
    body={
        # aggs：聚合查询（用于统计、分组、计算等，类似SQL的GROUP BY）
        "aggs": {
            "products_by_category": {  # 自定义的聚合名称
                # terms：分桶聚合（按指定字段的值分组，每个值对应一个“桶”，统计每个桶的文档数）
                "terms": {
                    "field": "category",  # 表示按 "category"（类别）字段分组
                    "size": 10  # 最多返回 10 个分组
                }
            }
        },
        "size": 0  # 不返回文档结果，只返回聚合结果
    }
)
print(json.dumps(res_3['aggregations']['products_by_category']['buckets'], ensure_ascii=False, indent=2))
