import datetime  # 日期时间处理（本文件未直接使用，可能为后续扩展预留）
# - BaseModel：所有数据模型的“基类”（相当于所有格式的“模板”）
# - Field：给字段添加描述、约束（比如说明这个字段是干嘛的）

# 从typing导入类型提示工具（明确字段的类型，比如“这是一个列表”“这可以是字符串或列表”）
from pydantic import BaseModel, Field

from typing import Union, List, Any, Tuple, Dict
# - Union[str, List[str]]：表示“可以是字符串，也可以是字符串列表”
# - List[str]：字符串列表；Tuple[str, str]：两个字符串组成的元组；Dict：字典

# 从FastAPI导入处理“表单数据”和“文件上传”的工具
from fastapi import FastAPI, File, UploadFile, Form
# - File：标记字段是“上传的文件”
# - UploadFile：表示“上传的文件对象”（包含文件名、内容等信息）
# - Form：标记字段是“表单中的普通数据”（比如输入框里的文本）

# 从typing_extensions导入Annotated（增强类型注解，明确字段的“来源”）
from typing_extensions import Annotated
# - 比如Annotated[str, Form()]：表示“这个字符串是从表单里来的”

"""
主要用于定义API接口的请求数据格式和响应数据格式
"""

# 语义嵌入请求
"""
示例：
{
  "text": ["这是一个测试句子", "另一个需要转换的文本"],
  "token": "user_auth_token_123",
  "model": "bge-small-zh-v1.5"
}
"""
class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]  # 输入文本：可以是单个字符串，或字符串列表（批量处理）
    token: str  # 用户认证令牌（类似“钥匙”，验证用户是否有权限调用接口）
    model: str  # 指定使用的嵌入模型（比如"bge-small-zh-v1.5"是一个中文嵌入模型）


# 语义嵌入响应
class EmbeddingResponse(BaseModel):
    request_id: str = Field(description="请求ID")  # 用Field添加描述，说明字段含义
    vector: List[List[float]] = Field(description="文本对应的向量表示")  # 向量列表（每个文本对应一个子列表）
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


# 重排序请求
"""
示例：
{
  "text_pair": [
    ("什么是RAG？", "RAG是检索增强生成的缩写"),
    ("什么是RAG？", "机器学习是人工智能的分支")
  ],
  "token": "user_auth_token_123",
  "model": "bge-reranker-base"
}
"""
class RerankRequest(BaseModel):
    text_pair: List[Tuple[str, str]]  # 文本对列表：每个元素是(查询文本, 候选文档文本)的元组
    token: str
    model: str  # 指定使用的重排序模型

# 重排序响应
class RerankResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    vector: List[float]  # 相关性分数列表：每个元素对应text_pair中一个文本对的分数（越高越相关）
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


# 知识库新增请求
"""
示例：
{
  "category": "技术手册",
  "title": "Python编程入门指南"
}
"""
class KnowledgeRequest(BaseModel):
    category: str   # 知识库分类（如"技术文档"、"产品手册"）
    title: str      # 知识库标题

# 知识库响应
class KnowledgeResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    knowledge_id: int   # 知识库ID（操作的目标知识库）
    category: str       # 知识库分类
    title: str          # 知识库标题
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


# 文档新增请求
class DocumentRequest(BaseModel):
    # 用Annotated明确字段来源：从表单获取（因为文件上传通常用表单格式）
    knowledge_id: int = Annotated[str, Form()],   # 所属知识库ID（从表单获取）
    title: str = Annotated[str, Form()],          # 文档标题（从表单获取）
    category: str = Annotated[str, Form()],       # 文档分类（从表单获取）
    file: UploadFile = Annotated[str, File(...)]  # 上传的文件（从表单文件域获取）

# 文档响应
class DocumentResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    document_id: int  # 文档ID（操作的目标文档）
    category: str     # 文档分类
    title: str        # 文档标题
    knowledge_id: int  # 所属知识库ID
    file_type: str     # 文件类型
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


# RAG聊天请求
"""
示例：
{
  "knowledge_id": 1001,
  "message": [
    {"role": "user", "content": "文档中提到的Python版本要求是什么？"}
  ]
}
"""
# class ChatMessage(BaseModel):
#     role: str
#     content: str

class RAGRequest(BaseModel):
    knowledge_id: int    # 目标知识库ID（指定基于哪个知识库回答）
    knowledge_title: str  # 知识库标题
    message: List[Dict]  # 聊天消息列表：每个元素是包含"role"（角色）和"content"（内容）的字典

# RAG聊天响应
class RAGResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    message: List[Dict]  # 回答消息列表：每个元素是包含"role"和"content"的字典（通常"role"为"assistant"）
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")
