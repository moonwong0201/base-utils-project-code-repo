import json
import jieba
import yaml  # type: ignore
from typing import Union, List, Any, Dict

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import numpy as np
import datetime
import pdfplumber  # 导入pdfplumber模块，用于处理PDF文件
from openai import OpenAI 

import torch  # type: ignore
# 加载预训练模型和分词器
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
# 加载嵌入模型
from sentence_transformers import SentenceTransformer  # type: ignore
# from FlagEmbedding import FlagReranker
from es_api import es  # 导入之前定义的Elasticsearch客户端（用于存储/检索数据）
from db_api import KnowledgeDatabase

"""
 RAG（检索增强生成）系统的核心实现，负责文档内容提取、文本嵌入生成、检索融合、与大语言模型（LLM）交互等关键功能
"""

device = config["device"]

# 存储嵌入模型和重排序模型的全局字典（加载后的数据存这里）
# 第一个 Any：代表字典的键 (key) 可以是任意类型
# 第二个 Any：代表字典的值 (value) 也可以是任意类型
EMBEDDING_MODEL_PARAMS: Dict[Any, Any] = {}

# RAG提示词模板（用于告诉LLM如何基于检索到的资料回答问题）
BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。
如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。

资料：
{#RELATED_DOCUMENT#}

问题：{#QUESTION#}
要求：回答必须标注内容对应的文档片段来源
'''


def load_embdding_model(model_name: str, model_path: str) -> None:
    """
    加载编码模型（将文本转为向量）
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    # sbert模型（中文常用嵌入模型，轻量且效果好）
    if model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        # 使用SentenceTransformer库加载模型，并存到全局字典
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(model_path)


# 加载重排序模型
def load_rerank_model(model_name: str, model_path: str) -> None:
    """
    加载重排序模型（对检索结果打分排序）
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    # 如果是BGE重排序模型
    if model_name in ["bge-reranker-base"]:
        # 加载重排序模型（分类任务模型，输出相关性分数）
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        # 加载对应的分词器（将文本转为模型能理解的token）
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        # 切换模型到评估模式（关闭dropout等训练相关层）
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(device)


#自动加载模型
# 如果配置中启用了嵌入功能（use_embedding: true）
if config["rag"]["use_embedding"]:
    # 获取当前使用的嵌入模型名称
    model_name = config["rag"]["embedding_model"]
    # 获取模型本地路径
    model_path = config["models"]["embedding_model"][model_name]["local_url"]

    print(f"Loading embedding model {model_name} from model_path...")
    load_embdding_model(model_name, model_path)

# 如果配置中启用了重排序功能（use_rerank: true）
if config["rag"]["use_rerank"]:
    model_name = config["rag"]["rerank_model"]
    model_path = config["models"]["rerank_model"][model_name]["local_url"]

    print(f"Loading rerank model {model_name} from model_path...")
    load_rerank_model(model_name, model_path)


# 文本分块函数
def split_text_with_overlap(text, chunk_size, chunk_overlap):
    """
    将长文本按指定大小分块，支持重叠部分（避免语义割裂）
    :param text: 待分块的文本
    :param chunk_size: 每个块的长度（单位：字符或token，这里简化为字符）
    :param chunk_overlap: 块之间的重叠长度
    :return: 分块后的文本列表
    """
    chunks = []
    start = 0  # 起始位置
    while start < len(text):
        end = start + chunk_size  # 结束位置
        chunk = text[start:end]  # 截取当前块
        chunks.append(chunk)
        # 下一个块的起始位置 = 当前起始 + 块大小 - 重叠大小（确保重叠）
        start = start + chunk_size - chunk_overlap        
    return chunks


"""
RAG类，整合了文档提取、向量生成、检索融合、LLM 交互所有功能
"""
class RAG:
    def __init__(self):
        # 从配置中获取当前使用的嵌入模型和重排序模型名称
        self.embedding_model = config["rag"]["embedding_model"]
        self.rerank_model = config["rag"]["rerank_model"]

        # 是否启用重排序（从配置读取）
        self.use_rerank = config["rag"]["use_rerank"]

        # 嵌入向量的维度
        self.embedding_dims = config["models"]["embedding_model"][
            config["rag"]["embedding_model"]
        ]["dims"]

        # 分块参数（从配置读取）
        self.chunk_size = config["rag"]["chunk_size"]            # 块大小
        self.chunk_overlap = config["rag"]["chunk_overlap"]      # 重叠大小
        self.chunk_candidate = config["rag"]["chunk_candidate"]  # 检索后保留的候选块数量

        # 初始化LLM客户端（连接大模型，支持OpenAI API格式的模型）
        self.client = OpenAI(
            api_key=config["rag"]["llm_api_key"],  # API密钥
            base_url=config["rag"]["llm_base"]     # API地址
        )
        self.llm_model = config["rag"]["llm_model"]  # 使用的LLM模型

        self.use_extract = config["rag"]["use_extract"]  # 是否调用关键词提取

    # PDF内容提取
    def _extract_pdf_content(self, knowledge_id, document_id, title, file_path) -> bool:
        # 提取PDF文件内容，并存储到ES
        try:
            # 用pdfplumber打开PDF文件
            pdf = pdfplumber.open(file_path)
        except:
            print("打开文件失败")
            return False

        print(f"{file_path} pages: ", len(pdf.pages))  # 打印提示信息，显示PDF文件的页数

        abstract = ""  # 用于存储文档摘要（前3页内容）

        for page_number in range(len(pdf.pages)):  # 每一页 提取
            current_page_text = pdf.pages[page_number].extract_text()  # 提取当前页的文本
            # 前3页文本合并为摘要（简单摘要生成逻辑）
            if page_number <= 3:
                abstract = abstract + '\n' + current_page_text

            # 1. 先存储整页内容作为一个块（chunk_id=0）
            # 生成整页文本的嵌入向量
            embedding_vector = self.get_embedding(current_page_text)
            # 构造存储到ES的数据（包含文档ID、知识库ID、页码、内容、向量等）
            page_data = {
                "document_id": document_id,    # 关联的文档ID
                "knowledge_id": knowledge_id,  # 关联的知识库ID
                "page_number": page_number,    # 所在页码
                "chunk_id": 0,        # 先存储每一页所有内容 0表示整页内容
                "chunk_content": current_page_text,   # 块内容（整页文本）
                "chunk_images": [],   # 存储图片信息（当前未实现）
                "chunk_tables": [],   # 存储表格信息（当前未实现）
                "embedding_vector": embedding_vector  # 整页文本的向量
            }
            # 存储到ES的chunk_info索引
            response = es.index(index="chunk_info", document=page_data)

            # 2. 将当前页文本分块，每个块单独存储（chunk_id从1开始）
            # 划分chunk
            page_chunks = split_text_with_overlap(current_page_text, self.chunk_size, self.chunk_overlap)
            # 批量生成所有分块的嵌入向量
            embedding_vector = self.get_embedding(page_chunks)
            # 遍历每个分块，存储到ES
            for chunk_idx in range(1, len(page_chunks) + 1):
                page_data = {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "page_number": page_number,
                    "chunk_id": chunk_idx,   # 表示分块编号
                    "chunk_content": page_chunks[chunk_idx - 1],  # 当前分块内容
                    "chunk_images": [],
                    "chunk_tables": [],
                    "embedding_vector": embedding_vector[chunk_idx - 1]  # 对应分块的向量
                }
                response = es.index(index="chunk_info", document=page_data)

        # 存储文档元数据到document_meta索引
        document_data = {
            "document_id": document_id,
            "knowledge_id": knowledge_id,
            "document_name": title,   # 文档标题
            "file_path": file_path,   # 文档本地路径
            "abstract": abstract      # 前3页生成的摘要
        }
        response = es.index(index="document_meta", document=document_data)

    # word内容提取
    def _extract_word_content(self):
        pass

    # 统一内容提取入口
    def extract_content(self, knowledge_id, document_id, title, file_type, file_path):
        # 根据文件类型调用对应的提取方法
        if "pdf" in file_type:    # 如果是PDF文件
            self._extract_pdf_content(knowledge_id, document_id, title, file_path)
        elif "word" in file_type:
            pass

        print("提取完成", document_id, file_type, file_path)

    # 生成文本嵌入向量
    def get_embedding(self, text) -> np.ndarray:
        """
        对文本进行编码，生成嵌入向量
        :param text: 待编码文本
        :return: 编码结果
        """
        # 如果使用BGE系列嵌入模型
        if self.embedding_model in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
            # 调用模型编码，normalize_embeddings=True表示向量归一化（方便计算余弦相似度）
            return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)

        # 如果使用其他模型，这里会抛出“未实现”的异常
        raise NotImplemented

    # 重排序打分
    def get_rank(self, text_pair) -> np.ndarray:
        """
        对查询-文档进行相关性打分（重排序）
        :param text_pair: 待排序文本 文本对列表，每个元素是[查询文本, 候选文档文本]
        :return: 匹配打分结果
        """
        if self.rerank_model in ["bge-reranker-base"]:
            with torch.no_grad():
                # 对文本对进行分词（转换为模型可处理的格式）
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair,
                    padding=True,      # 短文本补全到最长文本长度
                    truncation=True,   # 长文本截断到模型最大长度（如512）
                    return_tensors='pt',   # 返回PyTorch张量
                    max_length=512,   # 最大长度限制
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                # 模型推理，获取打分结果
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores

        raise NotImplemented

    # 文档检索
    def query_document(self, query: str, knowledge_id: int) -> List[str]:
        # 1. 全文检索，指定一个知识库检索，bm25打分
        # 向ES的"chunk_info"索引发送查询，指定“知识库ID”和“关键词匹配”
        word_search_response = es.search(
            index="chunk_info",
            body={
                "query": {
                    "bool": {   # 组合条件查询
                        "must": [  # 必须满足：chunk_content包含query的关键词
                            {
                                "match": {
                                    "chunk_content": query
                                }
                            }
                        ],
                        "filter": [  # 过滤条件：属于指定知识库
                            {
                                "term": {
                                    "knowledge_id": knowledge_id
                                }
                            }
                        ]
                    }
                },
                "size": 50  # 返回前50条结果
            },
            # 只返回需要的字段（减少数据传输）
            fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
            source=False,  # 不返回原始文档，只返回指定fields
        )

        # 2. 语义检索（kNN算法）：基于向量相似度，找和查询向量最像的文档块
        # 先把用户查询转成向量
        embedding_vector = self.get_embedding(query)  # 查询文本的嵌入向量 编码
        # 构造kNN检索参数
        knn_query = {
            "field": "embedding_vector",  # 检索的向量字段
            "query_vector": embedding_vector,  # 查询向量
            "k": 50,   # 返回最相似的50条
            "num_candidates": 100,  # 初步计算得到top100的待选文档，筛选最相关的50个
            "filter": {   # 只从指定知识库检索
                "term": {
                    "knowledge_id": knowledge_id
                }
            }
        }
        # 发送kNN检索请求
        vector_search_response = es.search(
            index="chunk_info",
            knn=knn_query,   # 使用kNN检索
            fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
            source=False,
        )

        # rrf
        # 检索1 ：[a， b， c]
        # 检索2 ：[b， e， a]
        # a 1/60    b 1/61    c 1/62
        # b 1/60    e 1/61    a 1/62

        # 3. RRF融合（Reciprocal Rank Fusion）：融合两种检索结果，提升准确性
        # RRF原理：每个检索结果的“排名”越靠前，贡献的分数越高，公式是 1/(排名 + k)
        k = 60   # RRF参数（控制排名权重）
        fusion_score = {}  # 存储每个文档块的融合分数
        search_id2record = {}  # 存储文档块ID到内容的映射

        # 处理全文检索结果，计算RRF分数
        """        
        word_search_response['hits']里面包含了与 “匹配文档” 相关的所有关键信息
        第二层 ['hits']是一个列表（List），每个元素是一个字典，代表一条匹配到的文档
        每条文档的结构类似这样：
        {
          "_index": "chunk_info",    // 文档所在的索引（chunk_info）
          "_id": "123456",           // 文档在ES中的唯一ID 同一个chunk是一样的
          "_score": 1.23,            // 这条文档与查询的相关性分数（BM25算法计算）
          "fields": {                // 通过fields参数指定要返回的字段
            "chunk_id": [0],
            "document_id": [1001],
            "knowledge_id": [5],
            "page_number": [3],
            "chunk_content": ["这是文档中的一段内容..."]
          }
        }
        """
        # 默认是按照 _score（相关性分数）从高到低排序的
        for idx, record in enumerate(word_search_response['hits']['hits']):    
            _id = record["_id"]  # 文档块在ES中的唯一ID
            # RRF分数公式：1/(排名+ k)，排名越靠前（idx越小），分数越高
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)  # 累加分数（同一文档块在多检索中出现）

            # 存储文档块内容（避免重复存储）
            if _id not in search_id2record:
                search_id2record[_id] = record["fields"]

        # 处理语义检索结果，计算RRF分数（逻辑同上）
        for idx, record in enumerate(vector_search_response['hits']['hits']):    
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)
            
            if _id not in search_id2record:
                search_id2record[_id] = record["fields"]

        # 4. 按融合分数排序，取前N个候选（chunk_candidate）
        # 对融合分数从高到低排序（sorted默认升序，reverse=True改为降序）
        sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
        # 取前chunk_candidate个结果
        sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]
        sorted_content = [x["chunk_content"] for x in sorted_records]

        # 5. 如果启用重排序，对候选结果再次打分排序
        if self.use_rerank:
            # 构造文本对：[查询, 候选文档块内容]
            text_pair = []
            for chunk_content in sorted_content:
                text_pair.append([query, chunk_content])
            rerank_score = self.get_rank(text_pair)  # 重排序打分
            # 按分数从高到低排序，获取索引
            rerank_idx = np.argsort(rerank_score)[::-1]

            # 按重排序后的索引更新结果
            # 排序后的文档块完整记录（含 ID、内容、页码等元数据）
            sorted_records = [sorted_records[x] for x in rerank_idx]
            # 排序后的文档块纯文本内容（从 sorted_records 中提取）
            sorted_content = [sorted_content[x] for x in rerank_idx]

        # 返回排序后的文档块详细信息（包含内容、页码、文档ID等）
        return sorted_records

    # RAG 聊天
    def chat_with_rag(
        self,
        knowledge_id: int,     # 知识库 哪一个知识库提问
        knowledge_title: str,  # 知识库标题
        messages: List[Dict],  # 聊天消息列表（包含用户问题）
        max_history: int = 5   # 最多保留 5 轮对话
    ):
        if len(messages) > max_history:
            # 保留最后 max_history 条（确保包含最新的用户问题）
            messages = messages[-max_history:]

        # 1. 单轮对话（只有用户的问题，没有历史上下文）
        if len(messages) == 1:
            query = messages[0]["content"]  # 提取用户问题

            # 是否要重写问题
            query = self.query_rewrite(query, "", knowledge_title)
            print(f"问题为：{query}")

            if self.use_extract:  # 是否要抽取
                query = self.query_parse(query)
                print(f"查询解析结果：{query}")

            related_records = self.query_document(query, knowledge_id)  # 检索到相关的文档
            # print(related_records)
            # 提取文档块内容，拼接为字符串
            # 这里的[0]，就是为了从列表中提取出那个唯一的字符串元素，从[...]变为...
            related_document = '\n'.join([x["chunk_content"][0] for x in related_records])

            # 填充提示词模板（替换时间、资料、问题）
            rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                .replace("{#QUESTION#}", query) \
                .replace("{#RELATED_DOCUMENT#}", related_document)

            # 调用LLM生成回答（基于资料）
            rag_response = self.chat(
                [{"role": "user", "content": rag_query}],
                0.7,     # top_p参数（控制输出多样性：0.9表示保留90%概率的词）
                0.9  # temperature参数（控制随机性：0.7适中，越高越随机）
            ).content
            # 把LLM的回答添加到消息列表（角色为"system"，表示系统回答）
            messages.append({"role": "system", "content": rag_response})
        # 2. 多轮对话（暂不结合知识库，直接调用LLM基于上下文回答）
        else:
            query = messages[-1]["content"]  # 拿到最新提问
            history_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-1]])

            # 是否要重写问题
            query = self.query_rewrite(query, history_content, knowledge_title)

            if self.use_extract:  # 是否要抽取
                query = self.query_parse(query)
                print(f"查询解析结果：{query}")

            related_records = self.query_document(query, knowledge_id)
            related_document = '\n'.join([x["chunk_content"][0] for x in related_records])

            rag_query = f"""历史上下文：{history_content}""" + BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                .replace("{#QUESTION#}", query) \
                .replace("{#RELATED_DOCUMENT#}", related_document)
            # 调用LLM，基于历史消息生成回答
            normal_response = self.chat(
                [{"role": "user", "content": rag_query}],  # 包含历史上下文（用户问题+之前的回答）
                0.7, 0.9
            ).content
            messages.append({"role": "system", "content": normal_response})

        # messages.append({"role": "system", "content": rag_response})
        # 返回包含“用户问题+系统回答”的消息列表
        return messages

    # 调用LLM
    def chat(self, messages: List[Dict], top_p: float, temperature: float) -> Any:
        try:
            # 调用OpenAI格式的LLM API，生成回答
            completion = self.client.chat.completions.create(
                model=self.llm_model,    # 使用的LLM模型
                messages=messages,       # 消息列表（包含角色和内容）
                top_p=top_p,             # 采样参数（控制输出多样性，0.9表示保留90%概率的词）
                temperature=temperature  # 温度参数（越高越随机，0.7适中）
            )
            # 返回LLM生成的第一条回答（通常只取第一个候选）
            return completion.choices[0].message
        except Exception as e:
            print(f"LLM调用失败：{e}")
            return type('obj', (object,), {'content': '抱歉，暂时无法回答，请稍后重试。'})

    # 查询解析：识别用户问题的意图、实体（暂未实现）
    def query_parse(self, query: str) -> Dict[str, str]:
        try:
            parse_prompt = f"""
            请分析以下用户问题，提取核心信息：
            1. 意图：用户想做什么？（如“查询用法”“询问部署步骤”“对比优缺点”）
            2. 实体：问题中的核心名词（如“RAG”“Python”“列表推导式”）
            3. 关键词：最核心的3-5个词（用逗号分隔）
            用户问题：{query}
            要求：只返回JSON格式，不要额外解释，示例：
            {{"intent": "查询用法", "entities": "RAG", "keywords": "RAG, 用法"}}
            """
            parse_response = self.chat(
                messages=[{"role": "user", "content": parse_prompt}],
                temperature=0.0,  # 0温度保证结果稳定
                top_p=0.1
            ).content
            parse_result = json.loads(parse_response)
            return parse_result
        except Exception as e:
            print(f"LLM解析失败，使用jieba兜底：{e}")
            keywords = jieba.lcut(query.strip())

            return {
                "intent": "意图未知",
                "entities": ",".join([k for k in keywords if len(k) > 1]),
                "keywords": ",".join(keywords)
            }

    # 查询重写：把模糊的问题改得更精准（如“怎么用？”→“怎么用RAG系统生成回答？”，暂未实现）
    def query_rewrite(self, query: str, context: str = "", knowledge_title: str = "") -> str:
        if len(query.strip()) >= 5:
            return query

        rewrite_prompt = f"""
        请根据以下信息，将用户的简略问题重写为完整、精准的问题：
        1. 知识库主题：{knowledge_title if knowledge_title else "通用技术文档"}
        2. 对话上下文：{context if context else "无"}
        3. 用户原始问题：{query}
        要求：
        - 重写后的问题要完整，包含核心实体和意图；
        - 保持原意，不新增用户没问的内容；
        - 只返回重写后的问题，不要额外解释。
        """
        try:
            rewrite_response = self.chat(
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.1,  # 低温度保证稳定性
                top_p=0.2
            ).content
            return rewrite_response.strip() if rewrite_response.strip() else query

        except Exception as e:
            print(f"查询重写失败，是用原始问题：{e}")
            return query
