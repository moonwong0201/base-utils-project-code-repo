import yaml   # type: ignore
with open("config.yaml", "r") as f:  # 从配置文件读取服务参数（如端口号）
    config = yaml.safe_load(f)

import logging  # 新增：导入日志模块
# 配置日志格式（包含时间、日志级别、文件名、行号、日志内容）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("rag_api.log", encoding="utf-8"),  # 日志写入文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)  # 创建日志实例

import time
import numpy as np
import uuid   # 生成唯一的“请求ID”（方便追踪每个用户请求）
import datetime
import traceback  # 捕获并打印异常详情

# 导入Web服务相关库
import uvicorn  # 运行FastAPI服务的服务器
from typing_extensions import Annotated  # 类型注解增强（指定参数来源）
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks  # FastAPI核心组件
# - FastAPI：创建API服务的“框架”（快速写接口的工具）
# - File/UploadFile：处理文件上传（比如上传PDF文档）
# - Form：处理表单数据（比如文档标题、所属知识库ID）
# - BackgroundTasks：后台任务（处理耗时操作，比如解析PDF内容，用户不用等）

# 定义API请求/响应的数据格式（Schema）
from router_schemas import (   # 定义API的“请求/响应格式”（相当于快递单模板）
    EmbeddingRequest, EmbeddingResponse,  # 语义嵌入的请求/响应格式
    RerankRequest, RerankResponse,        # 重排序的请求/响应格式
    KnowledgeRequest, KnowledgeResponse,  # 知识库的请求/响应格式
    DocumentRequest, DocumentResponse,    # 文档的请求/响应格式
    RAGRequest, RAGResponse               # RAG聊天的请求/响应格式
)
from rag_api import RAG  # 核心RAG功能（语义嵌入、检索、生成回答等）
from db_api import (  # 数据库操作相关类
    KnowledgeDocument, KnowledgeDatabase,   # 数据库“表结构”（相当于Excel的表头，定义存哪些字段）
    Session  # 数据库会话（用于操作数据库）
)

app = FastAPI()  # 创建FastAPI应用实例

"""
主要实现了知识库管理、文档管理和RAG相关功能
"""

# 查询知识库
# 根据knowledge_id（知识库唯一 ID）查询知识库的基本信息（标题、分类等）
@app.get("/v1/knowledge_base")
# 接口函数：参数是“要查的知识库ID”（knowledge_id）
# -> KnowledgeResponse：指定接口返回的格式（必须符合router_schemas里定义的模板）
def get_knowledge_base(knowledge_id: int) -> KnowledgeResponse:
    start_time = time.time()  # 记录接口开始时间（用于计算处理耗时）

    try:
        # 重试10次查询数据库（防止临时网络/数据库连接问题）
        for retry_time in range(10):
            with Session() as session:  # 创建数据库会话（自动管理连接，结束后关闭）
                # 数据库查询：从“KnowledgeDatabase表”里，找“knowledge_id等于传入参数”的第一条记录
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
                if record is not None:   # 如果找到这条记录（知识库存在）
                    # 返回符合“KnowledgeResponse”格式的结果
                    return KnowledgeResponse(     # type: ignore
                        request_id=str(uuid.uuid4()),   # 生成唯一请求ID（用于追踪）
                        knowledge_id=knowledge_id,      # 要查的知识库ID
                        title=str(record.title),        # 知识库标题
                        category=str(record.category),  # 知识库分类
                        response_code=200,              # 成功状态码
                        response_msg="知识库查询成功",    # 提示信息
                        process_status="completed",     # 处理状态
                        processing_time=time.time() - start_time  # 计算耗时
                    )
    except Exception as e:  # 如果过程中出错（比如数据库连不上）
        logger.error(f"查询知识库失败！knowledge_id={knowledge_id}, error={traceback.format_exc()}")

    # 如果没找到知识库（或出错），返回“不存在”的结果
    return KnowledgeResponse(  # type: ignore
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category="",   # 没找到，所以分类为空
        title="",      # 没找到，所以标题为空
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 删除知识库
@app.delete("/v1/knowledge_base")
def delete_knowledge_base(knowledge_id: int) -> KnowledgeResponse:
    start_time = time.time()

    try:
        for retry_time in range(10):  # 重试10次（防临时故障）
            with Session() as session:
                # 先查要删的知识库是否存在
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
                if record is None:  # 没找到，直接退出重试
                    break

                session.delete(record)  # 执行删除操作（从会话中标记为删除）
                session.commit()  # 提交事务（真正执行删除）
                return KnowledgeResponse(    ## type: ignore  返回删除成功响应
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    category=str(record.category),
                    title=str(record.title),
                    response_code=200,
                    response_msg="知识库删除成功",
                    process_status="completed",
                    processing_time=time.time() - start_time
                )
    except Exception as e:
        logger.error(f"删除知识库失败！knowledge_id={knowledge_id}, error={traceback.format_exc()}")

    # 未找到知识库，返回失败响应
    return KnowledgeResponse(  # type: ignore
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category="",
        title="",
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 新增知识库
# 创建一个新的知识库（需要提供标题和分类）
@app.post("/v1/knowledge_base")
def add_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    start_time = time.time()
    try:
        for retry_time in range(10):
            with Session() as session:
                # 创建“知识库记录”（对应数据库表的一行数据）
                record = KnowledgeDatabase(
                    title=req.title,        # 从请求中获取标题
                    category=req.category,  # 从请求中获取分类
                    create_dt=datetime.datetime.now(),  # 当前时间作为创建时间
                    update_dt=datetime.datetime.now(),  # 当前时间作为更新时间
                )
                session.add(record)  # 将记录添加到数据库会话
                # 刷新会话（触发数据库生成自增的knowledge_id）
                session.flush()  # Flushes changes to generate primary key if using autoincrement
                knowledge_id = record.knowledge_id  # 从刷新后的记录里拿“自动生成的ID”
                session.commit()  # 提交事务（真正保存到数据库）

            # 返回新增成功响应
            return KnowledgeResponse(  # type: ignore
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,  # 返回自动生成的ID
                category=req.category,
                title=req.title,
                response_code=200,
                response_msg="知识库插入成功",
                process_status="completed",
                processing_time=time.time() - start_time
            )

    except Exception as e:
        # print(traceback.format_exc())
        logger.error(f"新增知识库失败！title={req.title}, category={req.category}, error={traceback.format_exc()}")

    # 失败时返回插入失败响应
    return KnowledgeResponse(  # type: ignore
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        category="",
        title="",
        response_code=504,
        response_msg="知识库插入失败",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 文档管理接口
# 根据document_id查询文档的基本信息（标题、所属知识库、文件类型等）
@app.get("/v1/document")
def get_document(document_id: int) -> DocumentResponse:
    start_time = time.time()

    try:
        for retry_time in range(10):
            with Session() as session:
                # 查询指定document_id的文档记录
                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()

                if record is not None:  # 找到文档
                    return DocumentResponse(   # type: ignore
                        request_id=str(uuid.uuid4()),
                        document_id=document_id,
                        category=record.category,
                        title=record.title,
                        knowledge_id=record.knowledge_id,  # 所属知识库ID
                        file_type=record.file_type,  # 文件类型（如pdf、docx）
                        response_code=200,
                        response_msg="文档查询成功",
                        process_status="completed",
                        processing_time=time.time() - start_time
                    )
                break
    except Exception as e:
        # print(traceback.format_exc())
        logger.error(f"查询文档失败！document_id={document_id}, error={traceback.format_exc()}")

    # 未找到文档，返回失败响应
    return DocumentResponse(   # type: ignore
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg="文档不存在",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 删除文档
# 根据document_id删除指定文档
@app.delete("/v1/document")
def delete_document(document_id: int) -> DocumentResponse:
    start_time = time.time()

    try:
        for retry_time in range(10):
            with Session() as session:
                # 查询要删除的文档
                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()
                if record is None:
                    break

                session.delete(record)  # 标记删除
                session.commit()        # 提交删除
                return DocumentResponse(    # type: ignore
                    request_id=str(uuid.uuid4()),
                    document_id=document_id,
                    knowledge_id=record.knowledge_id,
                    category=record.category,
                    title=record.title,
                    file_type=record.file_type,
                    response_code=200,
                    response_msg="文档删除成功",
                    process_status="completed",
                    processing_time=time.time() - start_time
                )
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"删除文档失败！document_id={document_id}, error={traceback.format_exc()}")
    
    return DocumentResponse(   # type: ignore
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg="文档不存在",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 添加文档：判断知识库是否存在、解析上传的文件保存到本地、后台解析上传文件的内容
# 异步函数（async）：处理文件上传这类I/O操作更高效
# 上传一个文档并添加到指定知识库（核心接口之一）
# BackgroundTasks 后台任务执行
@app.post("/v1/document")
async def add_document(
    knowledge_id: int = Annotated[int, Form()],     # 从表单获取所属知识库ID
    title: str = Annotated[str, Form()],            # 从表单获取文档标题
    category: str = Annotated[str, Form()],         # 从表单获取文档分类
    file: UploadFile = Annotated[UploadFile, File(...)],   # 从表单获取上传的文件  File(...)表示“必须传文件”
    background_tasks: BackgroundTasks = BackgroundTasks()  # 后台任务工具 异步的体现
) -> DocumentResponse:
    start_time = time.time()
    response_msg ="新增文档失败"
    try:
        for retry_time in range(10):
            with Session() as session:
                # 第一步：先检查“文档要放的知识库是否存在”（不能往不存在的知识库传文件）
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
                if record is None:
                    response_msg = "知识库不存在，请提前创建"
                    break

                # 第二步：创建“文档记录”（存文档的基本信息，还没存文件内容）
                record = KnowledgeDocument(
                    title=title,
                    category=category,
                    knowledge_id=knowledge_id,  # 关联到对应的知识库
                    file_path="",   # 暂时为空，后面存完文件再更新
                    file_type=file.content_type,   # 文件类型（如application/pdf）
                    create_dt=datetime.datetime.now(),
                    update_dt=datetime.datetime.now(),
                )
                session.add(record)
                # 生成document_id
                session.flush()  # Flushes changes to generate primary key if using autoincrement
                document_id = record.document_id
                session.commit()

                # 第三步：保存上传的文件到本地（比如存到upload_files文件夹）
                # 文件名规则：document_id_原文件名（避免重名，比如两个都叫“手册.pdf”，用ID区分）
                file_path = f"upload_files/document_id_{document_id}_" + file.filename
                with open(file_path, "wb") as buffer:  # “wb”=二进制写入（文件都是二进制）
                    buffer.write(file.file.read())  # 读取上传的文件内容，写入本地文件

                # 第四步：更新“文档记录”的file_path（把本地文件路径存到数据库）
                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()
                record.file_path = file_path  # 更新路径
                session.commit()  # 提交更新

            # 第五步：后台处理“文件内容提取”（耗时操作，比如解析PDF里的文字，不用让用户等）
            background_tasks.add_task(
                RAG().extract_content,  # 后台运行的函数名（提取文件内容并生成嵌入）
                knowledge_id=knowledge_id,  # 以下都是函数的参数
                document_id=document_id,
                title=title,
                file_type=file.content_type,
                file_path=file_path
            )

            # 返回新增成功响应
            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                category=category,
                title=title,
                knowledge_id=knowledge_id,
                file_type=file.content_type,
                response_code=200,
                response_msg="文档添加成功",
                process_status="completed",
                processing_time=time.time() - start_time
            )
    except Exception as e:
        # print(traceback.format_exc())
        logger.error(f"新增文档失败！knowledge_id={knowledge_id}, title={title}, error={traceback.format_exc()}")

    return DocumentResponse(   # type: ignore
        request_id=str(uuid.uuid4()),
        document_id=0,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg=response_msg,
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 语义嵌入接口
# 将输入的文本（可以是单个句子或多个句子）转换为语义嵌入向量（数值数组）
@app.post("/v1/embedding")
async def semantic_embedding(req: EmbeddingRequest) -> EmbeddingResponse:
    start_time = time.time()
    # 确保输入文本是列表（如果是单个字符串，转为长度1的列表）
    if not isinstance(req.text, list):
        text = [req.text]
    else:
        text = req.text

    # 调用RAG的方法生成文本的语义嵌入向量
    vector: np.ndarray = RAG().get_embedding(text)
    # 返回结果：把numpy数组转成列表（因为数组不能直接转JSON，列表可以）
    return EmbeddingResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),  # 向量转为列表返回（方便JSON序列化）
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 重排序接口
# 对 “文本对”（比如 “查询文本” 和 “候选文档片段”）进行相关性评分，用于优化检索结果排序
@app.post("/v1/rerank")
async def semantic_rerank(req: RerankRequest) -> RerankResponse:
    start_time = time.time()
    # 调用RAG模块的“重排序”函数，输入“文本对列表”（比如[["RAG怎么用？", "RAG的使用步骤是..."], ...]）
    # 输出“分数数组”（每个文本对的相关性分数）
    vector: np.ndarray = RAG().get_rank(req.text_pair)

    return RerankResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),  # 分数转为列表返回
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# RAG聊天接口
# 结合指定知识库（knowledge_id）的内容，回答用户的问题（message），这是 RAG 系统的核心功能
@app.post("/chat")
def chat(req: RAGRequest) -> RAGResponse:
    start_time = time.time()
    # 调用RAG模块的“聊天函数”：传入“知识库ID”（从哪个知识库找内容）和“用户问题”
    # 返回“基于知识库生成的回答”
    message = RAG().chat_with_rag(req.knowledge_id, req.knowledge_title, req.message)
    
    return RAGResponse(
        request_id=str(uuid.uuid4()),
        message=message,   # 生成的回答
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time
    )


if __name__ == "__main__":
    # 用uvicorn运行FastAPI服务
    uvicorn.run(
        app,  # 要运行的API实例
        host="127.0.0.1",  # 允许所有设备访问（比如同一局域网的电脑能访问）
        port=config["rag"]["port"],  # 端口号从配置文件拿（比如8000）
        workers=1  # 用1个进程运行（简单场景够用，高并发时可加多个）
    )
