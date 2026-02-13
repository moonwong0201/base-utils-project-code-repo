"""
负责和 Elasticsearch (ES) 建立连接，并创建专门存储文档元数据和向量的索引结构
"""
import yaml  # type: ignore
from elasticsearch import Elasticsearch  # type: ignore  # ES的Python客户端（用于操作ES）
import traceback  # 用于捕获和打印详细的错误信息（方便排查问题）
from pathlib import Path  # 导入 pathlib

# 构建 config.yaml 的绝对路径
#    __file__ 变量包含了当前文件 (es_api.py) 的绝对路径
#    .parent 获取父目录，即项目根目录
#    / "config.yaml" 拼接成完整的配置文件路径
CONFIG_PATH = Path(__file__).parent / "config.yaml"

# 使用绝对路径打开文件
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 读取配置文件
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

"""
初始化ES连接并创建符合RAG需求的索引结构，确保后续能正常存储和检索数据
"""

# 从配置文件中提取ES的连接参数
es_host = config["elasticsearch"]["host"]          # ES服务器的主机地址
es_port = config["elasticsearch"]["port"]          # ES的端口号（默认9200）
es_scheme = config["elasticsearch"]["scheme"]      # 连接协议（通常是"http"或"https"）
es_username = config["elasticsearch"]["username"]  # 认证用户名
es_password = config["elasticsearch"]["password"]  # 认证密码

# 根据是否有用户名密码，创建不同的ES客户端
if es_username != "" and es_password != "":
    # 有认证的情况：使用账号密码连接
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
        basic_auth=(es_username, es_password)  # 传入认证信息
    )
else:
    # 无认证的情况：直接连接（适合本地测试环境）
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
    )

# 从配置中获取嵌入向量的维度（与嵌入模型匹配）
embedding_dims = config["models"]["embedding_model"][
    config["rag"]["embedding_model"]  # 从rag配置中获取当前使用的嵌入模型
]["dims"]  # 对应模型的向量维度


# 初始化ES环境
def init_es():
    """
    检查ES连接并创建必要的索引结构
    :return: 环境是否配置成功（True/False）
    """

    # 第一步：检查ES是否连接成功
    if not es.ping():  # ping()方法测试与ES的连接
        print("Could not connect to Elasticsearch.")
        return False

    # 第二步：创建document_meta索引（存储文档的基本信息，不是文档内容本身）
    # 定义document_meta索引的映射（mapping）：规定字段类型和分词方式
    document_meta_mapping = {
        "mappings": {    # mappings定义索引的字段结构
            'properties': {    # 定义索引中的所有字段
                'file_name': {      # 字段1：文件名
                    'type': 'text',  # 文本类型（可分词检索）
                    'analyzer': 'ik_max_word',   # 索引时分词器（ik_max_word：中文细粒度分词，适合长文本）
                    'search_analyzer': 'ik_max_word'   # 查询时分词器（与索引时保持一致，确保检索准确）
                },
                'abstract': {       # 字段2：文档摘要（简要描述文档内容）
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                },
                'full_content': {   # 字段3：文档完整内容（可选，大文档可能不存储）
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                }
            }
        }
    }
    try:
        # 检查索引是否已存在，不存在则创建
        # es.indices.delete(index='document_meta')
        if not es.indices.exists(index="document_meta"):   # 检查索引是否存在
            es.indices.create(index='document_meta', body=document_meta_mapping)  # 创建索引
    except:
        print(traceback.format_exc())  # 打印异常详情
        print("Could not create index of document_meta.")
        return False

    # 创建chunk_info索引（文档分块与向量索引）
    # 定义chunk_info索引的映射：存储分块内容和嵌入向量
    # 实际上最好所有字段都要有的，只不过其他简单的字段es能够猜测到，而chunk_content和embedding_vector是最核心且复杂的字段，让es猜测可能会有问题
    chunk_info_mapping = {
        'mappings': {  # Add 'mappings' here  文档分块内容（如将PDF按256token拆分后的片段）
            'properties': {
                'chunk_content': {     # 字段1：文档分块内容
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                },
                "embedding_vector": {  # 字段2：分块内容的语义嵌入向量
                    "type": "dense_vector",   # ES的向量类型（用于存储密集向量）
                    "element_type": "float",  # 向量元素类型（浮点型）
                    "dims": embedding_dims,   # 向量维度（与嵌入模型一致，如512）
                    "index": True,            # 启用索引（支持向量检索）
                    "index_options": {
                        "type": "int8_hnsw"   # 向量索引类型（hnsw算法，高效近似最近邻检索；int8压缩存储节省空间）
                    }
                }
            }
        }
    }

    try:
        # 检查索引是否存在，不存在则创建
        # es.indices.delete(index='chunk_info')  # 开发时删除旧索引用
        if not es.indices.exists(index="chunk_info"):
            es.indices.create(index='chunk_info', body=chunk_info_mapping)
    except:
        print(traceback.format_exc())
        print("Could not create index of chunk_info.")
        return False

    print("Successfully connected to Elasticsearch!")
    return True


init_es()
