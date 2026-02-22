import json
import os
import random
import string
from datetime import datetime, UTC
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from agents.mcp import MCPServerSse, ToolFilterStatic

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable
from fastapi.responses import StreamingResponse

from config import (
    OPENAI_API_KEY, OPENAI_VISION_MODEL, OPENAI_MODEL, OPENAI_BASE_URL,
    DB_PATH
)

# """
# 聊天业务的核心服务层，承接 routers/chat.py 接口的请求，
# 整合了「会话管理、AI Agent 调度、工具调用、数据库持久化」四大核心能力，
# 是连接 “前端聊天接口” 和 “底层 AI / 数据库” 的关键桥梁
# """


# 作用：为新对话生成唯一的 session_id（对话标识），保证每个用户的每轮对话有独立标识；
# 核心逻辑：生成随机字符串后，查询 ChatSessionTable 校验唯一性，重试 20 次仍重复则抛异常；
# 使用场景：用户首次发起聊天时，调用该函数生成 session_id，用于后续多轮对话上下文关联。
def generate_random_chat_id(length=12):
    with SessionLocal() as session:
        for retry_time in range(20):
            # 生成12位字母+数字的随机session_id
            characters = string.ascii_letters + string.digits
            session_id = ''.join(random.choice(characters) for i in range(length))
            # 校验是否已存在（避免重复）
            chat_session_record: ChatSessionTable | None = session.query(ChatSessionTable).filter(
                ChatSessionTable.session_id == session_id).first()
            if chat_session_record is None:
                return session_id

            if retry_time == 19:
                raise Exception("Failed to generate a unique session_hash")

    raise Exception("Unexpected error in generate_random_chat_id")


# 作用：根据用户指定的 task（对话任务），动态生成个性化的 AI 系统提示词；
# 使用场景：初始化对话时，为 AI Agent 提供 “身份 + 任务约束”，确保 AI 回复符合场景要求（如股票分析用专业术语，通用对话更友好）。
def get_init_message(
        task: str,
) -> str:
    try:
        # 加载Jinja2模板（chat_start_system_prompt.jinja2）
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("chat_start_system_prompt.jinja2")
    except TemplateNotFound:
        # 模板不存在时返回默认提示词
        default_prompt = "你是一个智能助手，请根据用户的问题提供专业、友好的回答。"
        return default_prompt

    # 按任务类型（股票分析/数据BI/通用对话）定制任务描述
    if task == "股票分析":
        task_description = """
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
5. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
"""
    elif task == "数据BI":
        task_description = """
1. 帮助用户理解他们的数据结构、商业指标和关键绩效指标 (KPI)。
2. 用户的请求通常是数据查询、指标定义或图表生成建议。
3. **关键约束：你的输出必须是可执行的代码块 (如 SQL 或 Python)，或者清晰的逻辑步骤，用于解决用户的数据问题。**
4. 严格遵守数据分析的逻辑严谨性，确保每一个结论都有数据支撑。
5. 当被要求提供可视化建议时，请推荐最合适的图表类型（如：时间序列用折线图，分类对比用柱状图）。"""
    else:
        task_description = """
1. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
2. 避免过于专业或生硬的术语，除非用户明确要求。
3. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
4. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
5. 关键词：友好、轻松、富有同理心。
        """

    # 渲染模板，生成最终系统提示词
    system_prompt = template.render(
        agent_name="小呆助手",
        task_description=task_description,
        current_datetime=datetime.now(UTC),
    )
    return system_prompt


# 为新对话初始化数据库记录，完成 “会话元数据 + 系统提示词” 的持久化
# 使用场景：用户首次发起聊天（无 session_id）时，调用该函数创建会话记录。
def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        task: str,
) -> bool:

    # 创建对话的title，通过summary agent
    # 存储数据库
    with SessionLocal() as session:
        # 1. 根据用户名查用户ID（关联UserTable）
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return False

        # 2. 写入ChatSessionTable（会话元数据）
        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question[: 100],  # 会话标题=用户首个问题
            start_time=datetime.now(UTC)
        )
        print("add ChatSessionTable", user_id[0], session_id)
        session.add(chat_session_record)
        session.commit()
        session.flush()

        init_message = get_init_message(task)

        # 3. 写入ChatMessageTable（系统提示词，作为第一条消息）
        message_record = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=init_message,
            create_time=datetime.now(UTC)
        )
        session.add(message_record)
        session.flush()
        session.commit()

    return True


async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    # 对话管理，通过session id
    # 步骤1：会话校验（无session_id则初始化，有则校验是否存在）
    if not session_id:  # 无session_id时需要生成并初始化
        session_id = generate_random_chat_id()
        init_chat_session(user_name, content, session_id, task or "通用对话")  # 初始化数据库记录，不需要接收返回值
    else:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 步骤2：存储用户消息到关系型数据库
    append_message2db(session_id, "user", content)

    # 步骤3：获取system message系统提示词，需要传给大模型，并不能给用户展示
    instructions = get_init_message(task or "通用对话")

    # 步骤4：初始化OpenAI客户端（对接阿里云通义千问）
    external_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    # 步骤5：MCP工具过滤（指定可用工具列表）
    # 如果用户没勾选工具：tool_filter=None（Agent 不调用任何工具）；
    # 如果用户勾选了工具：用 ToolFilterStatic 限定 Agent 只能调用勾选的工具（比如只允许查股票，不允许查新闻）。
    if not tools or len(tools) == 0:  # 前端选择的工具
        tool_mcp_tools_filter: Optional[ToolFilterStatic] = None
    else:
        tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools)

    try:
        mcp_server = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            tool_filter=tool_mcp_tools_filter,
            client_session_timeout_seconds=20,
        )
    except Exception as e:
        yield f"MCP服务初始化失败：{str(e)}"  # yield是把错误信息 “流式推送给前端”，而不是一次性返回
        return

    # openai-agent支持的session存储，存储对话的历史状态
    # 步骤6：初始化Agent会话记忆（AdvancedSQLiteSession）
    try:
        session = AdvancedSQLiteSession(  # 存的是 “给 AI 看的上下文”（极简，只保留角色 + 内容），供 Agent 做多轮对话时参考，不关联业务字段
            session_id=session_id,  # 与系统中的对话 id 关联，存储在关系型数据库中
            db_path=DB_PATH,
            create_tables=True
        )
    except Exception as e:
        yield f"会话记忆初始化失败：{str(e)}"
        return

    # 分支1：不调用工具 → 直接调用大模型流式回答
    if not tools or len(tools) == 0:
        try:
            agent = Agent(
                name="Assistant",
                instructions=instructions,  # 系统提示词
                model=OpenAIChatCompletionsModel(
                    model=OPENAI_MODEL,
                    openai_client=external_client,
                ),
                # tool_use_behavior="stop_on_first_tool",
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            # 流式调用Agent
            result = Runner.run_streamed(agent, input=content, session=session)

            assistant_message = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent):  # 如果是大模型的回答
                        if event.data.delta:
                            yield f"{event.data.delta}"  # 流式返回给前端（SSE）
                            assistant_message += event.data.delta

            # 存储AI回复到数据库
            append_message2db(session_id, "assistant", assistant_message)
        except Exception as e:
            error_msg = f"大模型调用失败：{str(e)}"
            yield error_msg
            append_message2db(session_id, "assistant", error_msg)

    # 分支2：调用MCP工具 → 先调用工具，再返回结果（或让大模型总结）
    else:
        try:
            async with mcp_server:
                # 判定工具类型：可视化工具（直接返回结果）/其他工具（大模型总结）
                need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
                if set(need_viz_tools) & set(tools):
                    tool_use_behavior = "stop_on_first_tool"  # 调用了tool，得到结果，就展示结果
                else:
                    tool_use_behavior = "run_llm_again"  # 调用了tool，得到结果，继续用大模型的总结结果

                # 初始化带工具的Agent
                agent = Agent(
                    name="Assistant",
                    instructions=instructions,
                    mcp_servers=[mcp_server],  # 关联MCP工具服务
                    model=OpenAIChatCompletionsModel(
                        model=OPENAI_MODEL,
                        openai_client=external_client,
                    ),
                    tool_use_behavior=tool_use_behavior,  # 工具调用策略
                    model_settings=ModelSettings(parallel_tool_calls=False)
                )
                # 流式调用Agent（含工具）
                result = Runner.run_streamed(agent, input=content, session=session)

                assistant_message = ""
                current_tool_name = ""
                async for event in result.stream_events():
                    # if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output" and current_tool_name not in need_viz_tools:
                    #     yield event.item.raw_item["output"]
                    #     assistant_message += event.item.raw_item["output"]

                    # tool_output

                    # 工具调用结果：返回工具名+参数（JSON格式）
                    if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                        if isinstance(event.data.item, ResponseFunctionToolCall):
                            current_tool_name = event.data.item.name

                            # 工具名字、工具参数
                            yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                            assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

                    # 大模型总结结果：流式返回文本
                    if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                        yield event.data.delta
                        assistant_message += event.data.delta

                # 存储AI+工具的回复到数据库
                append_message2db(session_id, "assistant", assistant_message)
        except Exception as e:
            error_msg = f"工具调用失败：{str(e)}"
            yield error_msg
            append_message2db(session_id, "assistant", error_msg)


# 根据 session_id 查询该会话下的所有消息（用户 / AI / 系统），返回给前端展示对话历史
def get_chat_sessions(session_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        chat_messages: Optional[List[ChatMessageTable]] = session.query(ChatMessageTable) \
            .join(ChatSessionTable, ChatMessageTable.chat_id == ChatSessionTable.id) \
            .filter(ChatSessionTable.session_id == session_id) \
            .order_by(ChatMessageTable.create_time.asc()) \
            .all()

        result = []
        if chat_messages:
            for record in chat_messages:
                result.append({
                    "id": record.id,
                    "create_time": record.create_time,
                    "feedback": record.feedback,
                    "feedback_time": record.feedback_time,
                    "role": record.role,
                    "content": record.content,
                    "generated_sql": record.generated_sql,   # 预留存储 AI 生成的 SQL 语句
                    "generated_code": record.generated_code  # 预留存储 AI 生成的代码
                })

        return result


# 根据 session_id 删除整个会话（先删 ChatMessageTable 消息，再删 ChatSessionTable 会话）
def delete_chat_session(session_id: str) -> bool:
    with SessionLocal() as session:
        chat_session_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if chat_session_id is None:
            return False

        try:
            session.query(ChatMessageTable).where(ChatMessageTable.chat_id == chat_session_id[0]).delete()
            session.query(ChatSessionTable).where(ChatSessionTable.id == chat_session_id[0]).delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()  # 失败回滚
            print(f"删除会话失败：{e}")
            return False


# 为单条消息添加用户反馈（满意 / 不满意），存入 ChatMessageTable.feedback
def change_message_feedback(session_id: str, message_id: int, feedback: bool) -> bool:
    with SessionLocal() as session:
        chat_session_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if chat_session_id is None:
            return False

        record = session.query(ChatMessageTable).filter(
            ChatMessageTable.id == message_id,
            ChatMessageTable.chat_id == chat_session_id[0]
        ).first()
        if record is not None:
            try:
                record.feedback = feedback
                record.feedback_time = datetime.now(UTC)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"修改反馈失败：{e}")
                return False

        return False


# 	根据用户名查询该用户的所有会话列表（用于前端展示 “我的对话”）
def list_chat(user_name: str) -> List[ChatSession]:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if user_id:
            chat_records: List[ChatSessionTable] = session.query(ChatSessionTable) \
                                         .filter(ChatSessionTable.user_id == user_id[0]) \
                                         .order_by(ChatSessionTable.start_time.desc()) \
                                         .all()
            if chat_records:
                return [ChatSession(
                    user_id=x.user_id,
                    session_id=x.session_id,
                    title=x.title,
                    start_time=x.start_time
                ) for x in chat_records]
            else:
                return []
        else:
            return []


# 将用户 / AI 消息写入 ChatMessageTable，完成消息持久化
def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        message_record = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if message_record:
            try:
                message_record = ChatMessageTable(
                    chat_id=message_record[0],
                    role=role,
                    content=content
                )
                session.add(message_record)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"写入消息失败：{e}")
                return False
        return False
