import json
import os
import random
import string
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, AsyncGenerator

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from agents.mcp import MCPServerSse
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable

from config import (
    OPENAI_API_KEY, OPENAI_VISION_MODEL, OPENAI_MODEL, OPENAI_BASE_URL,
    DB_PATH
)
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generate_random_chat_id(length=12):
    with SessionLocal() as session:
        for retry in range(20):
            session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            exists = session.query(ChatSessionTable).filter_by(session_id=session_id).first()
            if not exists:
                return session_id
        raise Exception("Failed to generate unique session_id after 20 retries")


def get_init_message(task: str, available_tools: List[str] = None) -> str:
    try:
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("chat_start_system_prompt.jinja2")
    except TemplateNotFound:
        return "你是一个智能助手，请根据用户的问题提供专业、友好的回答。"

    if task == "stock":
        task_description = f"""
你是专业的股票分析助手，仅处理股票相关问题，禁止闲聊无关内容。
核心能力：
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
5. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
6. **你只能使用以下工具：{available_tools or '所有股票工具'}，禁止调用其他工具！**
"""
    elif task == "通用对话":
        task_description = """
1. 你是一个闲聊小助手，擅长应对各种闲聊话题，禁止讨论股票相关内容。
2. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
3. 避免过于专业或生硬的术语，除非用户明确要求。
4. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
5. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
6. 关键词：友好、轻松、富有同理心。
"""
    else:
        task_description = "你是一个智能助手，请根据用户的问题提供专业、友好的回答。"

    system_prompt = template.render(
        agent_name="小呆助手",
        task_description=task_description,
        current_datetime=datetime.now(),
    )
    return system_prompt


def create_chat_agent(openai_client: AsyncOpenAI) -> Agent:
    """创建闲聊 Agent（无工具）"""
    return Agent(
        name="chat_agent",
        instructions=get_init_message("通用对话"),
        model=OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=openai_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def create_stock_agent(
        openai_client: AsyncOpenAI,
        mcp_server: MCPServerSse,  # ← 接受外部共享的 server
        tools: List[str],
        tool_use_behavior: str
) -> Agent:
    """创建股票 Agent，使用共享 MCP Server"""

    # 关键：在 instructions 里约束可用工具，替代 tool_filter
    available_tools_str = ", ".join(tools) if tools else "所有股票工具"

    return Agent(
        name="stock_agent",
        instructions=get_init_message("stock",
                                      tools) + f"\n\n**严格约束：你只能使用这些工具：[{available_tools_str}]，禁止调用其他工具！**",
        mcp_servers=[mcp_server],  # ← 共享 server
        model=OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=openai_client,
        ),
        tool_use_behavior=tool_use_behavior,
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def create_other_agent(
        openai_client: AsyncOpenAI,
        mcp_server: MCPServerSse,  # ← 接受外部共享的 server
        tools: List[str]
) -> Agent:
    """创建资讯生活 Agent，使用共享 MCP Server"""

    available_tools_str = ", ".join(tools) if tools else "所有资讯工具"

    instructions = f"""
你是资讯生活助手，擅长：
- 查询新闻热点（头条、抖音、GitHub、体育）
- 提供每日名言、励志语录、心灵鸡汤  
- 生活查询（天气、地址、手机号归属、景点、花语、汇率）

保持回答简洁有趣，适合日常阅读。
禁止讨论股票相关内容。

**严格约束：你只能使用这些工具：[{available_tools_str}]，禁止调用其他工具！**
"""

    return Agent(
        name="other_agent",
        instructions=instructions,
        mcp_servers=[mcp_server],  # ← 共享 server
        model=OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=openai_client,
        ),
        tool_use_behavior="run_llm_again",
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def create_triage_agent(openai_client: AsyncOpenAI, agents: List[Agent]) -> Agent:
    """创建调度 Agent"""
    triage_instructions = """
你是任务调度专家，必须将用户请求转交给合适的专家处理，禁止自己回答。

【强制规则】
1. 分析用户输入，选择最合适的专家
2. 使用 handoff 功能，立即将请求转交给该专家
3. 绝对禁止自己直接回答

【选择规则】
- 含股票/股价/行情/茅台/腾讯/大盘/K线/财报 → handoff 给 stock_agent
- 含新闻/天气/名言/热点/抖音/汇率/景点 → handoff 给 other_agent
- 其他所有情况 → handoff 给 chat_agent

如果不确定，一律 handoff 给 chat_agent，禁止自己回答。
"""
    return Agent(
        name="triage_agent",
        instructions=triage_instructions,
        model=OpenAIChatCompletionsModel(
            model=OPENAI_MODEL,
            openai_client=openai_client,
        ),
        handoffs=agents,
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def init_chat_session(user_name: str, user_question: str, session_id: str, task: str) -> bool:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return False

        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question[: 100],
            start_time=datetime.now(UTC)
        )
        session.add(chat_session_record)
        session.commit()
        session.flush()

        init_message = get_init_message(task)

        message_record = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=init_message,
            create_time=datetime.now(UTC)
        )
        session.add(message_record)
        session.commit()

        logger.info(f"会话初始化成功: {session_id}, task: {task}")
    return True


async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    logger.info(f"=== chat 启动 === 用户: {user_name}, 输入: {content[:50]}")

    # 步骤1：会话初始化
    if not session_id:
        session_id = generate_random_chat_id()
        init_chat_session(user_name, content, session_id, task or "通用对话")
    else:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    append_message2db(session_id, "user", content)

    # 步骤2：初始化 OpenAI 客户端
    external_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    # 步骤3：初始化会话记忆
    try:
        session = AdvancedSQLiteSession(
            session_id=session_id,
            db_path=DB_PATH,
            create_tables=True
        )
    except Exception as e:
        yield f"data: 会话记忆初始化失败：{str(e)}\n\n"
        return

    logger.info("=== Multi-Agent分支 ===")

    # 统一创建 MCP Server，所有 Agent 共享
    mcp_server = MCPServerSse(
        name="Shared MCP Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,  # 不加 tool_filter，所有工具暴露
        client_session_timeout_seconds=30,
    )

    try:
        # 手动连接
        logger.info("=== 连接 MCP Server ===")
        await mcp_server.connect()  # 手动连接
        logger.info("=== MCP Server 连接成功 ===")

        # 准备工具列表
        STOCK_TOOLS = {"get_stock_code", "get_index_code", "get_industry_code",
                       "get_board_info", "get_stock_rank", "get_month_line", "get_week_line",
                       "get_day_line", "get_stock_info", "get_stock_minute_data"}
        INFO_TOOLS = {"get_today_daily_news", "get_douyin_hot_news", "get_github_hot_news",
                      "get_toutiao_hot_news", "get_sports_news", "get_today_famous_saying",
                      "get_today_motivation_saying", "get_today_working_saying", "get_city_weather",
                      "get_address_detail", "get_tel_info", "get_scenic_info", "get_flower_info",
                      "get_rate_transform"}

        # stock_tools = list(set(tools) & STOCK_TOOLS)
        # other_tools = list(set(tools) & INFO_TOOLS)
        stock_tools = list(STOCK_TOOLS)
        other_tools = list(INFO_TOOLS)

        logger.info(f"stock_tools: {stock_tools}")
        logger.info(f"other_tools: {other_tools}")

        # 判定工具行为
        need_viz_tools = {"get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"}
        tool_use_behavior = "stop_on_first_tool" if (set(tools) & need_viz_tools) else "run_llm_again"

        # 创建所有 Agent，共享同一个 mcp_server
        chat_agent = create_chat_agent(external_client)
        stock_agent = create_stock_agent(external_client, mcp_server, stock_tools, tool_use_behavior)
        other_agent = create_other_agent(external_client, mcp_server, other_tools)

        agents = [chat_agent, stock_agent, other_agent]
        logger.info(f"创建 Triage, agents: {[a.name for a in agents]}")

        triage_agent = create_triage_agent(external_client, agents)
        tool_calls_seen: set = set()
        # 运行
        logger.info("=== 开始 Runner.run_streamed ===")
        result = Runner.run_streamed(triage_agent, input=content, session=session)

        logger.info("=== 进入事件循环 ===")
        assistant_message = ""
        tool_displayed = False

        async for event in result.stream_events():
            logger.info(f"事件: {event.type}")

            # ===== 1. 处理 run_item_stream_event =====
            if event.type == "run_item_stream_event":
                item = event.item
                item_type = type(item).__name__
                logger.info(f"run_item 类型: {item_type}")

                # 1.1 真正工具调用：ToolCallItem
                if item_type == 'ToolCallItem':
                    tool_name = getattr(item, 'name', '')
                    tool_args = getattr(item, 'arguments', '{}')

                    if tool_name and tool_name not in tool_calls_seen:
                        tool_block = f"\n```json\n{tool_name}:{tool_args}\n```\n\n"
                        yield f"data: {tool_block}\n\n"
                        tool_calls_seen.add(tool_name)
                        logger.info(f"✅ 工具调用显示: {tool_name}")
                    continue

                # 1.2 handoff 相关：跳过
                if item_type in ['HandoffCallItem', 'HandoffOutputItem']:
                    logger.info(f"Handoff 事件，跳过")
                    continue

                # 1.3 其他（MessageOutputItem 等）：跳过
                continue

            # ===== 2. 处理 raw_response_event =====
            elif event.type == "raw_response_event":
                data = event.data

                # 2.1 文本输出
                if isinstance(data, ResponseTextDeltaEvent):
                    if data.delta:
                        yield f"data: {data.delta}\n\n"
                        assistant_message += data.delta

                # 2.2 工具调用完成（备用，如果 ToolCallItem 没抓到）
                elif isinstance(data, ResponseOutputItemDoneEvent):
                    if isinstance(data.item, ResponseFunctionToolCall):
                        tool_name = data.item.name
                        if tool_name not in tool_calls_seen and 'transfer_to_' not in tool_name:
                            tool_payload = {
                                "type": "tool_call",
                                "name": tool_name,
                                "arguments": data.item.arguments
                            }
                            tool_text = json.dumps(tool_payload, ensure_ascii=False)
                            yield f"data: {tool_text}\n\n"
                            tool_calls_seen.add(tool_name)
                            logger.info(f"工具调用显示(备用): {tool_name}")

        append_message2db(session_id, "assistant", assistant_message)
        logger.info(f"Multi-Agent 分支完成，输出长度: {len(assistant_message)}")

    except Exception as e:
        logger.error(f"工具调用失败: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"工具调用失败：{str(e)}"
        yield f"data: {error_msg}\n\n"
        append_message2db(session_id, "assistant", error_msg)

    finally:
        # 手动断开连接
        try:
            await mcp_server.disconnect()
            logger.info("=== MCP Server 断开 ===")
        except Exception as e:
            logger.warning(f"MCP 断开失败: {e}")


def get_chat_sessions(session_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        chat_messages = session.query(ChatMessageTable) \
            .join(ChatSessionTable, ChatMessageTable.chat_id == ChatSessionTable.id) \
            .filter(ChatSessionTable.session_id == session_id) \
            .order_by(ChatMessageTable.create_time.asc()) \
            .all()

        return [{
            "id": r.id,
            "create_time": r.create_time,
            "feedback": r.feedback,
            "feedback_time": r.feedback_time,
            "role": r.role,
            "content": r.content,
        } for r in chat_messages]


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
            session.rollback()
            print(f"删除会话失败：{e}")
            return False


def change_message_feedback(session_id: str, message_id: int, feedback: bool) -> bool:
    with SessionLocal() as session:
        chat_session_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if chat_session_id is None:
            return False

        record = session.query(ChatMessageTable).filter(
            ChatMessageTable.id == message_id,
            ChatMessageTable.chat_id == chat_session_id[0]
        ).first()

        if record:
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


def list_chat(user_name: str) -> List[ChatSession]:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if user_id:
            chat_records = session.query(ChatSessionTable) \
                .filter(ChatSessionTable.user_id == user_id[0]) \
                .order_by(ChatSessionTable.start_time.desc()) \
                .all()
            return [ChatSession(
                user_id=r.user_id,
                session_id=r.session_id,
                title=r.title,
                start_time=r.start_time
            ) for r in chat_records]
        return []


def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        chat_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if chat_id:
            try:
                message = ChatMessageTable(
                    chat_id=chat_id[0],
                    role=role,
                    content=content
                )
                session.add(message)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"写入消息失败：{e}")
                return False
        return False
