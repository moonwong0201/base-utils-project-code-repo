import traceback
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from agents import Agent, OpenAIChatCompletionsModel, Runner
# Agent：AI 智能体基类，封装了 AI 的思考、工具调用、对话管理等能力。
# OpenAIChatCompletionsModel：OpenAI 聊天补全模型的封装类，用于调用 OpenAI 的聊天接口（如 gpt-3.5-turbo、gpt-4 等）。
# Runner：智能体运行器，负责调度 AI 智能体执行任务（如处理用户消息、调用工具等）。

from agents.extensions.memory import AdvancedSQLiteSession  # 用于聊天会话的持久化存储
from fastapi import FastAPI, APIRouter, Query  # type: ignore
from fastapi.responses import StreamingResponse  # 实现流式响应（SSE），支持实时向客户端推送聊天消息
from typing import AsyncGenerator, Union, List, Optional
import os  # Need to import os for environment variables

from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

import services.chat as chat_services
from models.data_models import (
    BasicResponse,
    RequestForChat,
    ResponseForChat,
    ChatSession
)

router = APIRouter(prefix="/v1/chat", tags=["chat"])
# prefix="/v1/chat"：所有接口的 URL 前缀，比如 /v1/chat/（发送消息）、/v1/chat/init（初始化会话），统一接口路径规范。
# tags=["chat"]：在 FastAPI 自动生成的 API 文档（/docs）中，将这些接口归为 “chat” 分组，便于前端开发者查找和测试。

# """
# 智能问答功能的路由层核心，也是前端 chat.py 和后端服务层 services/chat.py 之间的 “中转站”
# —— 它只做「接口定义、参数校验、响应封装」，不碰核心业务逻辑，是请求从前端到服务层的必经之路
#
# 定义聊天功能的所有后端接口，统一路径和响应格式；
# 做基础参数校验，拦截无效请求，保护服务层；
# 封装流式响应，实现 AI 回复的实时推送；
# 所有核心业务逻辑都转发给服务层，自身只做 “中转站”。
# """


# 整个智能问答的核心接口，前端 chat.py 发起的聊天请求都走这里：
# 参数校验：先检查 user_name 和 content 是否为空（空则返回 400 错误，且封装成 SSE 流式格式，保证前端能统一处理）；
# 调用服务层：把前端传的参数（用户名、会话 ID、内容、勾选的工具列表）传给 chat_services.chat()（服务层核心函数）；
# 流式响应封装：将服务层返回的异步数据流，封装成 SSE（Server-Sent Events）格式（data: {内容}\n\n），实时返回给前端，实现 “边生成边展示” 的打字效果；
# 异常兜底：无论参数错误、服务层异常，都返回流式响应（而非普通 JSON），保证前端接收逻辑统一。
# 装饰器：将下面的 chat 函数注册为 POST 请求接口，URL 路径为 /v1/chat/，用于接收用户聊天消息并返回 AI 的流式响应
@router.post("/", response_model=None)
async def chat(req: RequestForChat) -> StreamingResponse:
    if not req.user_name.strip():
        # 流式响应异常处理：返回符合SSE格式的错误信息
        async def error_generator():
            yield f"data: {BasicResponse(code=400, message='用户名不能为空', data={}).model_dump_json()}\n\n"

        return StreamingResponse(error_generator(), media_type="text/event-stream", status_code=400)

    if not req.content.strip():
        async def error_generator():
            yield f"data: {BasicResponse(code=400, message='聊天内容不能为空', data={}).model_dump_json()}\n\n"

        return StreamingResponse(error_generator(), media_type="text/event-stream", status_code=400)

    try:
        # 异步函数 用于生成流式响应的数据块
        async def chat_stream_generator():
            try:
                # async for 遍历异步数据流
                async for chunk in chat_services.chat(
                        user_name=req.user_name,
                        task=req.task or "通用对话",
                        session_id=req.session_id,
                        content=req.content,
                        tools=req.tools or []
                ):
                    # 每次迭代获取一个数据块并立即 yield 返回
                    # 将 chat_services.chat 生成的每个数据块实时返回给客户端，实现 “边生成边展示” 的流式效果
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                # 流式响应内部异常处理
                error_msg = f"MCP服务初始化失败：{str(e)}"
                yield f"data: {BasicResponse(code=500, message=error_msg, data={}).model_dump_json()}\n\n"

        # Server-Sent Events (SSE) sse 对话流式输出，实时数据流
        return StreamingResponse(
            content=chat_stream_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        error_msg = f"聊天接口初始化失败：{str(e)}"
        print(traceback.format_exc())
        # 异常时仍返回流式响应（符合前端统一处理逻辑）
        async def global_error_generator():
            yield f"data: {BasicResponse(code=500, message=error_msg, data={}).model_dump_json()}\n\n"
        return StreamingResponse(global_error_generator(), media_type="text/event-stream", status_code=500)


# 作用：给前端生成唯一的 session_id（用于关联多轮对话）；
# 逻辑：调用 chat_services.generate_random_chat_id() 生成 ID，封装成标准响应返回。
@router.post("/init", response_model=BasicResponse[dict])
async def init_chat() -> BasicResponse:
    try:
        session_id = chat_services.generate_random_chat_id()
        return BasicResponse(
            code=200,
            message="会话初始化成功",
            data={"session_id": session_id}
        )
    except Exception as e:
        error_msg = f"会话初始化失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data={})


# 作用：前端 chat.py 加载历史对话时调用；
# 逻辑：接收 session_id，调用 chat_services.get_chat_sessions() 获取历史记录，封装后返回。
@router.post("/get", response_model=BasicResponse[List[ResponseForChat]])
def get_chat(session_id: str = Query(..., description="会话ID")) -> BasicResponse:
    logger.info(f"收到 session_id={session_id}")

    if not session_id or not session_id.strip():
        logger.warning("会话ID为空")
        return BasicResponse(code=400, message="会话ID不能为空", data=[])

    try:
        chat_records = chat_services.get_chat_sessions(session_id)
        logger.info(f"查询到 {len(chat_records)} 条记录")

        if chat_records:
            logger.info(f"第一条记录 keys: {chat_records[0].keys()}")

        if not chat_records:
            return BasicResponse(code=200, message="该会话暂无聊天记录", data=[])

        # 尝试构造返回数据
        response = BasicResponse(
            code=200,
            message="查询聊天记录成功",
            data=chat_records
        )
        logger.info("构造响应成功")
        return response

    except Exception as e:
        logger.error(f"查询失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return BasicResponse(code=500, message=f"查询失败: {str(e)}", data=[])


# session_id: str：请求参数，接收客户端传递的会话 ID（用于定位要查询的聊天记录）
# def get_chat(session_id: str = Query(..., description="会话ID")) -> BasicResponse:
#     if not session_id or session_id == "None" or not session_id.strip():
#         return BasicResponse(code=400, message="会话ID不能为空", data=[])
#
#     try:
#         chat_records = chat_services.get_chat_sessions(session_id)
#         if not chat_records:
#             return BasicResponse(code=200, message="该会话暂无聊天记录", data=[])
#         return BasicResponse(
#             code=200, message="查询聊天记录成功",
#             data=chat_records
#         )
#     except Exception as e:
#         error_msg = f"查询聊天记录失败：{str(e)}"
#         print(traceback.format_exc())
#         return BasicResponse(code=500, message=error_msg, data=[])


# 删除指定会话（包括会话关联的聊天记录）
@router.post("/delete", response_model=BasicResponse[bool])
def delete_chat(session_id: str = Query(..., description="会话ID")) -> BasicResponse:
    if not session_id.strip():
        return BasicResponse(code=400, message="会话ID不能为空", data=False)

    try:
        delete_result = chat_services.delete_chat_session(session_id)
        if delete_result:
            return BasicResponse(code=200, message="会话删除成功", data=delete_result)
        else:
            return BasicResponse(code=404, message="会话不存在或删除失败", data=False)
    except Exception as e:
        error_msg = f"删除会话失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=False)


# 查询指定用户的所有聊天会话列表
@router.post("/list", response_model=BasicResponse[List[ChatSession]])
def list_chat(user_name: str = Query(..., description="用户名")) -> BasicResponse:
    if not user_name.strip():
        return BasicResponse(code=400, message="用户名不能为空", data=[])

    try:
        chat_records = chat_services.list_chat(user_name)
        if not chat_records:
            return BasicResponse(code=200, message="该用户暂无聊天会话", data=[])
        return BasicResponse(code=200, message="查询会话列表成功", data=chat_records)
    except Exception as e:
        error_msg = f"查询会话列表失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=[])


# 提交聊天消息的反馈
@router.post("/feedback", response_model=BasicResponse[bool])
def feedback_chat(session_id: str, message_id: int, feedback: bool) -> BasicResponse:
    """
    提交对单条聊天消息的反馈
    :param session_id: 会话 ID（关联哪个会话的消息）
    :param message_id: 消息 ID（具体反馈哪一条消息）。
    :param feedback: 反馈结果（True 表示有用，False 表示无用）。
    :return: 反馈提交结果
    """
    if not session_id.strip():
        return BasicResponse(code=400, message="会话ID不能为空", data=False)
    if message_id <= 0:
        return BasicResponse(code=400, message="消息ID必须为正整数", data=False)

    try:
        feedback_result = chat_services.change_message_feedback(session_id, message_id, feedback)
        if feedback_result:
            return BasicResponse(code=200, message="反馈提交成功", data=True)
        else:
            return BasicResponse(code=404, message="会话/消息不存在", data=False)

    except Exception as e:
        print(traceback.format_exc())
        error_msg = f"提交反馈失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=False)
