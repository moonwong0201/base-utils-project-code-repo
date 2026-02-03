"""
用户打开页面 → 初始化会话状态/聊天历史 → 侧边栏配置（Token/模型/工具开关）
→ 用户输入问题 → 保存到聊天历史 → 调用get_model_response3（模式3）
→ 分发代理（triage_agent）语义理解 → 分发给对应子代理（比如新闻→News Assistant）
→ 子代理调用MCP后端工具 → 工具返回数据 → AI整理数据生成回答
→ 前端流式显示（工具调用信息+AI回答）→ 保存回答到聊天历史
"""

# ----------------------------
# 1. 导入依赖库（基础工具+项目核心模块）
# ----------------------------
import traceback  # 捕获并打印错误堆栈，方便调试（比如工具调用失败时看具体原因）
from datetime import datetime  # 获取当前时间，给日志加时间戳（方便定位问题发生时间）

import streamlit as st  # 核心：构建前端交互界面（聊天框、侧边栏、消息展示）
from agents.mcp.server import MCPServerSse  # 核心：MCP服务器客户端（连接后端8900端口的工具服务）
import asyncio  # 处理异步任务（工具调用、流式响应都是异步操作，不阻塞界面）
from agents import (  # 导入AI代理相关核心类（openai-agent库封装的功能）
    Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession, RunConfig, ModelSettings
)
# 导入OpenAI流式响应的类型定义（用于识别不同类型的响应事件，比如文本片段、工具调用）
from openai.types.responses import (
    ResponseTextDeltaEvent, ResponseCreatedEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
)
# 导入MCP工具过滤相关类（静态过滤、动态过滤，控制子代理能调用哪些工具）
from agents.mcp import MCPServer, ToolFilterStatic, ToolFilterCallable
# OpenAI-agent库的全局配置：设置默认API类型+关闭追踪（减少冗余日志，提高性能）
from agents import set_default_openai_api, set_tracing_disabled

# ----------------------------
# 2. 全局初始化配置（项目启动时执行一次）
# ----------------------------
# 设置openai-agent默认使用"chat_completions"类型API（适配通义千问的聊天接口）
set_default_openai_api("chat_completions")
# 关闭agent的追踪功能（避免打印过多内部日志，让终端只显示关键信息）
set_tracing_disabled(True)

# 配置Streamlit页面基础信息：页面标题（浏览器标签栏显示）
st.set_page_config(page_title="企业职能机器人")

# 创建SQLite会话（用于保存AI代理的对话上下文，内存级存储，重启后清空）
# 参数"conversation_123"是会话ID，不同ID可区分不同对话
session = SQLiteSession("conversation_123")

# ----------------------------
# 3. 构建侧边栏（用户配置区：Token输入、模型选择、功能开关）
# ----------------------------
with st.sidebar:  # Streamlit的侧边栏上下文（所有缩进内的内容都会显示在侧边栏）
    st.title('职能AI+智能问答')  # 侧边栏标题（大标题，突出主题）

    # 逻辑：判断用户是否已配置通义千问的API Token（避免重复输入）
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        # 已配置：显示绿色成功提示+✅图标
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']  # 从会话状态读取已保存的Token
    else:
        # 未配置：默认填充一个测试Token（可替换成自己的）
        key = "sk-399b434c3f5b4329a4600ec76ce4f7cc"

    # 侧边栏添加密码输入框：让用户输入/修改通义千问API Token（type='password'隐藏输入内容）
    key = st.text_input('输入Token:', type='password', value=key)
    # 把用户输入的Token保存到Streamlit会话状态（刷新页面不丢失，全局可用）
    st.session_state['API_TOKEN'] = key

    # 侧边栏添加下拉框：让用户选择AI模型（qwen-flash快/轻量，qwen-max准/ heavy）
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    # 侧边栏添加复选框：控制是否启用工具调用（勾选=调用后端MCP工具，不勾选=纯AI聊天）
    use_tool = st.checkbox("使用工具")


# ----------------------------
# 4. 初始化+渲染聊天历史（保存对话上下文，让AI能记住之前的对话）
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

# 渲染历史对话：遍历"messages"列表，按角色（system/user/assistant）显示聊天内容
for message in st.session_state.messages:
    # st.chat_message()：Streamlit的聊天消息组件，自动区分用户/助手样式（左对齐/右对齐）
    with st.chat_message(message["role"]):
        st.write(message["content"])  # 显示消息具体内容


# ----------------------------
# 5. 清空聊天历史功能（重置对话上下文）
# ----------------------------
def clear_chat_history():
    # 1. 重置会话状态中的聊天消息：恢复到默认欢迎语（清空前端显示的历史）
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]
    # 2. 声明使用全局变量session（因为要修改外部定义的SQLite会话）
    global session
    # 3. 重新创建SQLite会话：清空AI代理的对话上下文（AI不再记住之前的对话）
    session = SQLiteSession("conversation_123")

# 侧边栏添加"清空聊天"按钮：点击时触发clear_chat_history函数
st.sidebar.button('清空聊天', on_click=clear_chat_history)


# ----------------------------
# 6. 核心逻辑1：3种工具调用模式（按需选择，当前用get_model_response3）
# ----------------------------
# 模式1：无过滤，所有工具对AI可见（适合工具少的场景）
async def get_model_response1(prompt, model_name, use_tool):
    """
    :param prompt: 用户输入的问题
    :param model_name: 选择的AI模型（qwen-flash/max）
    :param use_tool: 是否启用工具调用（True/False）
    :return: 生成器（流式返回事件类型+内容：argument/raw/content）
    """
    # 异步创建MCP服务器客户端（连接后端8900端口的MCP服务，工具都在后端注册）
    async with (MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},  # 后端MCP服务地址（必须和mcp_server_main.py配置一致）
        cache_tools_list=False,  # 不缓存工具列表（每次调用都重新获取后端工具，适合工具动态更新）
        client_session_timeout_seconds=20,  # 客户端超时时间（20秒没响应则失败）
    ) as mcp_server):  # async with：自动管理客户端连接（创建→使用→关闭，无需手动关闭）

        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 逻辑：根据是否启用工具，创建不同配置的AI代理
        if use_tool:
            # 启用工具：创建关联MCP服务器的代理（能调用后端所有工具）
            agent = Agent(
                name="Assistant",
                instructions="",  # 代理的系统指令（空=使用默认逻辑）
                mcp_servers=[mcp_server],  # 关联MCP服务器（关键：让代理能调用工具）
                model=OpenAIChatCompletionsModel(  # 配置AI模型
                    model=model_name,  # 选择用户指定的模型
                    openai_client=external_client,
                )
            )
        else:
            # 不启用工具：创建纯AI聊天代理（不能调用任何工具，直接回答）
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        # 运行AI代理：以流式方式处理用户输入（prompt），关联对话会话（session）
        # Runner.run_streamed()：openai-agent库的核心函数，返回流式事件生成器
        result = Runner.run_streamed(agent, input=prompt, session=session)

        # 异步遍历流式事件（逐个处理AI返回的事件）
        async for event in result.stream_events():
            print(datetime.now(), "111", event)

            # 事件1：检测工具调用请求（AI发起了工具调用）
            if (event.type == "raw_response_event"  # 事件类型：原始响应事件
                and hasattr(event, 'data')  # 事件有data属性
                and isinstance(event.data, ResponseOutputItemDoneEvent)  # data是输出项完成事件
            ):
                print(datetime.now(), "222", event)
                # 判断是否是工具调用（ResponseFunctionToolCall是工具调用的类型标识）
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    # 生成器返回：事件类型"argument" + 工具调用参数（供前端显示/调试）
                    yield "argument", event.data.item

            # 事件2：检测工具返回结果（后端工具调用完成，返回数据）
            if (event.type == "run_item_stream_event"  # 事件类型：运行项流式事件
                    and hasattr(event, 'name')  # 事件有name属性
                    and event.name == "tool_output"):  # name是"tool_output"=工具返回
                print(datetime.now(), "333", event)
                # 生成器返回：事件类型"raw" + 工具原始返回数据
                yield "raw", event.item.raw_item["output"]

            # 事件3：检测AI生成的文本（工具调用完成后，AI整理结果生成回答）
            if (event.type == "raw_response_event"  # 事件类型：原始响应事件
                    and hasattr(event, 'data')  # 有data属性
                    and isinstance(event.data, ResponseTextDeltaEvent)  # data是文本片段事件
            ):
                print(datetime.now(), "444", event)
                # 生成器返回：事件类型"content" + 文本片段（供前端流式显示）
                yield "content", event.data.delta


# 模式2：静态工具过滤（指定子代理只能调用特定工具，硬编码工具名）
async def get_model_response2(prompt, model_name, use_tool):
    # 静态过滤1：新闻子代理只能调用这2个工具（get_today_daily_news、get_github_hot_news）
    news_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(
        allowed_tool_names=["get_today_daily_news", "get_github_hot_news"]
    )
    # 静态过滤2：工具子代理只能调用这2个工具（get_city_weather、sentiment_classification）
    tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(
        allowed_tool_names=["get_city_weather", "sentiment_classification"]
    )

    # 创建新闻工具的MCP客户端（绑定新闻工具过滤）
    mcp_server2 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=news_mcp_tools_filter,  # 关联新闻工具过滤
        client_session_timeout_seconds=20,
    )

    mcp_server1 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    external_client = AsyncOpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 异步管理2个MCP客户端（同时连接新闻和通用工具服务）
    async with mcp_server1, mcp_server2:
        if use_tool:
            # 子代理1：新闻代理（专门处理新闻相关问题，只能调用新闻工具）
            news_agent = Agent(
                name="News Assistant",
                instructions="Solve task, like 查询新闻",  # 告诉代理擅长处理的任务
                mcp_servers=[mcp_server2],  # 关联新闻MCP客户端
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            # 子代理2：通用工具代理（专门处理天气/情感分析，只能调用通用工具）
            tool_agnet = Agent(
                name="Tool Assistant",
                instructions="Solve task, like 查询天气",  # 告诉代理擅长处理的任务
                mcp_servers=[mcp_server1],  # 关联通用MCP客户端
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            # 主代理：分发代理（triage_agent），负责根据用户问题分发给对应的子代理
            agent = Agent(
                name="triage_agent",
                # 系统指令：根据用户请求的语言/类型，分发给合适的子代理
                instructions="Handoff to the appropriate agent based on the language of the request.",
                handoffs=[news_agent, tool_agnet],  # 关联所有子代理（告诉主代理有哪些子代理可用）
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
        else:
            # 不启用工具：纯AI聊天代理（和模式1一致）
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        # 运行代理+处理流式事件
        result = Runner.run_streamed(
            agent, input=prompt, session=session,
            run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False))
        )

        async for event in result.stream_events():
            print(datetime.now(), "111", event)

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                print(datetime.now(), "222", event)
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    yield "argument", event.data.item

            if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output":
                print(datetime.now(), "333", event)
                yield "raw", event.item.raw_item["output"]

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(datetime.now(), "444", event)
                yield "content", event.data.delta


# 动态工具过滤函数1：新闻工具过滤（返回True=允许调用，False=过滤）
def mcp_news_callable_filter(context, tool) -> bool:
    # 只允许调用这2个新闻工具（和模式2的静态过滤逻辑一致，只是用函数实现）
    return tool.name == "get_today_daily_news" or tool.name == "get_github_hot_news"


# 动态工具过滤函数2：通用工具过滤
def mcp_tool_callable_filter(context, tool):
    # 只允许调用这2个通用工具
    return tool.name == "get_city_weather" or tool.name == "sentiment_classification"


# 模式3：动态工具过滤（用函数控制工具可见性，更灵活，当前项目使用这个模式）
async def get_model_response3(prompt, model_name, use_tool):
    mcp_server1 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_tool_callable_filter,  # 动态过滤：只允许通用工具
        client_session_timeout_seconds=20,
    )

    # 创建新闻工具的MCP客户端（绑定动态过滤函数mcp_news_callable_filter）
    mcp_server2 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_news_callable_filter,  # 动态过滤：只允许新闻工具
        client_session_timeout_seconds=20,
    )

    external_client = AsyncOpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 异步管理2个MCP客户端
    async with mcp_server1, mcp_server2:

        if use_tool:
            # 子代理1：新闻代理（和模式2一致，只能调用新闻工具）
            news_agent = Agent(
                name="News Assistant",
                instructions="Solve task, like 查询新闻",
                mcp_servers=[mcp_server2],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            # 子代理2：通用工具代理（和模式2一致，只能调用通用工具）
            tool_agnet = Agent(
                name="Tool Assistant",
                instructions="Solve task, like 查询天气",
                mcp_servers=[mcp_server1],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            # 主代理：分发代理（核心：基于语义理解分发给子代理）
            agent = Agent(
                name="triage_agent",
                instructions="Handoff to the appropriate agent based on the language of the request.",
                handoffs=[news_agent, tool_agnet],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )


        result = Runner.run_streamed(agent, input=prompt, session=session, run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)))

        async for event in result.stream_events():
            print(datetime.now(), "111", event)

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                print(datetime.now(), "222", event)
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    yield "argument", event.data.item

            if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output":
                print(datetime.now(), "333", event)
                yield "raw", event.item.raw_item["output"]

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(datetime.now(), "444", event)
                yield "content", event.data.delta


# ----------------------------
# 7. 核心逻辑2：前端聊天交互流程（用户输入→调用代理→流式显示结果）
# ----------------------------
# 逻辑：只有用户输入的API Token有效（长度>1），才允许发送聊天请求
if len(key) > 1:
    # st.chat_input()：创建底部聊天输入框，用户输入后返回输入内容（prompt）
    if prompt := st.chat_input():
        # 1. 保存用户输入到会话状态（更新聊天历史，让AI能看到之前的消息）
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 2. 渲染用户输入的消息（在前端显示用户的问题）
        with st.chat_message("user"):
            st.markdown(prompt)  # 用markdown格式显示（支持换行、链接等）

        # 3. 渲染助手的响应（流式显示AI回答+工具调用信息）
        with st.chat_message("assistant"):
            placeholder = st.empty()   # 创建空占位符（用于动态更新响应内容，实现流式效果）

            with st.spinner("请求中..."):
                try:
                    # 内部异步函数：遍历流式生成器，累积内容并更新前端
                    async def stream_output():
                        accumulated_text = ""  # 存储完整的响应文本
                        # 调用模式3的代理逻辑，获取流式事件生成器
                        response_generator = get_model_response3(prompt, model_name, use_tool)
                        # 异步遍历生成器（逐个处理事件）
                        async for event_type, chunk in response_generator:

                            # 事件类型1：工具调用参数（argument）→ 格式化显示为JSON代码块（方便调试）
                            if event_type == "argument":
                                # chunk 可能是 dict/list/其他对象 —— 转成字符串以防报错
                                formatted_raw = f"\n\n```json\n[RawArg]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw  # 累积内容
                                placeholder.markdown(accumulated_text + "▌")  # 显示累积内容+光标
                            # 事件类型2：工具返回结果（raw）→ 格式化显示为JSON代码块
                            elif event_type == "raw":
                                # chunk 可能是 dict/list/其他对象 —— 转成字符串以防报错
                                formatted_raw = f"\n\n```json\n[RawEvent]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")
                            # 事件类型3：AI生成的文本（content）→ 直接累积并显示
                            elif event_type == "content":
                                # chunk 应该是 str（文本片段）
                                accumulated_text += chunk
                                placeholder.markdown(accumulated_text + "▌")

                        return accumulated_text  # 返回完整响应文本

                    # 在同步上下文（Streamlit是同步的）中运行异步函数
                    final_text = asyncio.run(stream_output())
                    # 响应完成：移除光标，显示完整文本
                    placeholder.markdown(final_text)

                except Exception as e:
                    error_msg = f"发生错误: {e}"
                    placeholder.error(error_msg)  # 用红色错误样式显示
                    final_text = error_msg
                    traceback.print_exc()  # 打印错误堆栈（调试用，看具体哪里错了）

            # 4. 保存助手的响应到会话状态（更新聊天历史）
            st.session_state.messages.append({"role": "assistant", "content": final_text})


# import streamlit as st
# from agents.mcp.server import MCPServerSse, ToolFilterContext, MCPTool
# import asyncio
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
# # 导入自定义agents模块的核心类：
# # Agent：智能代理（协调AI模型和工具调用）；Runner：运行代理的执行器；AsyncOpenAI：异步OpenAI兼容客户端；
# # OpenAIChatCompletionsModel：ChatCompletion类型的AI模型封装；SQLiteSession：SQLite数据库会话（存储对话历史）
#
# from openai.types.responses import ResponseTextDeltaEvent  # 导入OpenAI类型定义，用于识别流式响应的文本片段事件
# from agents.mcp import MCPServer
# from agents import set_default_openai_api, set_tracing_disabled
# set_default_openai_api("chat_completions")
# set_tracing_disabled(True)
#
# st.set_page_config(page_title="职能机器人")
# session = SQLiteSession("conversation_123")  # openai-agent提供的基于内存的上下文缓存
#
# with st.sidebar:  # 创建页面侧边栏（用于放置配置项）
#     st.title('职能AI+智能问答')  # 侧边栏标题
#
#     # 检查会话状态中是否已保存API_TOKEN，且长度大于1（判断是否已配置Token）
#     # session_state 保存当前的对话缓存
#     if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
#         st.success('API Token已经配置', icon='✅')  # 已配置则显示成功提示
#         key = st.session_state['API_TOKEN']
#     else:
#         key = ""
#
#     # 侧边栏添加密码输入框，用于输入API Token，默认值为已保存的key
#     key = st.text_input('输入Token:', type='password', value=key)
#
#     st.session_state['API_TOKEN'] = key  # 将输入的Token保存到会话状态（刷新页面不丢失）
#     model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])  # 侧边栏添加下拉框，选择AI模型
#     use_tool = st.checkbox("使用工具")  # 侧边栏添加复选框，控制是否启用工具调用（连接后端MCP服务）
#
#
# # 初始化对话历史：如果会话状态中没有"messages"，则创建默认欢迎消息
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [
#         {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]
#
# # 渲染历史对话：遍历会话状态中的所有消息，按角色（user/assistant）显示
# # streamlit提供的：st.session_state.messages此次对话的历史上下文
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):  # 按角色创建聊天消息框（用户/助手区分样式）
#         st.write(message["content"])  # 显示消息内容
#
#
# # 定义清空聊天历史的函数
# def clear_chat_history():
#     # 重置会话状态中的消息，恢复默认欢迎语
#     st.session_state.messages = [
#         {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]
#
#     global session  # 声明使用全局变量session
#     session = SQLiteSession("conversation_123")  # 重新创建SQLite会话（清空对话数据库）
#
#
# # 侧边栏添加"清空聊天"按钮，点击时调用clear_chat_history函数
# st.sidebar.button('清空聊天', on_click=clear_chat_history)
#
#
# async def tools_filter(context: ToolFilterContext, tool: MCPTool):
#     # 直接从前端会话状态中提取最后一条用户输入
#     user_input = ""
#     # 遍历前端保存的所有消息（倒序找最新的用户输入）
#     for msg in reversed(st.session_state.messages):
#         if msg["role"] == "user":  # 匹配用户角色的消息
#             user_input = msg["content"].lower().strip()
#             break  # 找到第一条（最新的）用户消息后退出
#
#     if not user_input:
#         return False
#     else:
#         print(f"从前端会话中获取的用户输入：{user_input}")
#
#     tool_description = tool.description or ""
#     if "[type: news]" in tool_description:
#         tool_type = "news"
#     elif "[type: tool]" in tool_description:
#         tool_type = "tool"
#     elif "[type: saying]" in tool_description:
#         tool_type = "saying"
#     else:
#         tool_type = None
#
#     if not tool_type:
#         return False
#
#     if any(keyword in user_input for keyword in ["今日新闻", "热门", "话题", "抖音", "GitHub", "今日头条", "电竞", "体育"]):
#         is_match = tool_type == "news"
#         if is_match:
#             print(f"关键词匹配成功，{tool.name} 备选")
#         else:
#             print(f"关键词匹配失败，{tool.name} 过滤")
#         return is_match
#     elif any(keyword in user_input for keyword in ["天气", "地址", "电话", "旅游", "景点", "花", "货币", "汇率", "情感", "分析"]):
#         is_match = tool_type == "tool"
#         if is_match:
#             print(f"关键词匹配成功，{tool.name} 备选")
#         else:
#             print(f"关键词匹配失败，{tool.name} 过滤")
#         return is_match
#     elif any(keyword in user_input for keyword in ["名言", "一言", "励志", "激励", "职场", "心灵鸡汤"]):
#         is_match = tool_type == "saying"
#         if is_match:
#             print(f"关键词匹配成功，{tool.name} 备选")
#         else:
#             print(f"关键词匹配失败，{tool.name} 过滤")
#         return is_match
#
#     print(f"无匹配关键词，{tool.name} 过滤")
#     return False
#
#
# async def get_model_response(prompt, model_name, use_tool):
#     """
#     :param prompt: 当前用户输入
#     :param model_name: 模型版本
#     :param use_tool: 是否调用工具
#     :return:
#     """
#     # 异步创建MCP的SSE客户端，连接后端MCP服务器（之前主程序启动的8900端口SSE服务）
#     async with MCPServerSse(
#             name="SSE Python Server",  # MCP客户端名称
#             params={
#                 "url": "http://localhost:8900/sse",  # 后端MCP服务器的SSE访问地址
#             },
#             client_session_timeout_seconds=20,
#             tool_filter=tools_filter
#     )as mcp_server:  # 上下文管理器自动管理MCP客户端的连接/关闭
#
#         # 创建异步OpenAI兼容客户端（对接阿里云通义千问API）
#         external_client = AsyncOpenAI(
#             api_key=key,
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#         )
#         # 根据是否启用工具，创建不同配置的Agent（智能代理）
#         if use_tool:
#             agent = Agent(
#                 name="Assistant",
#                 instructions="",   # 代理的系统指令
#                 mcp_servers=[mcp_server],  # 关联MCP服务器（启用工具调用，可调用news/saying/tool模块的功能）
#                 model=OpenAIChatCompletionsModel(  # 配置AI模型
#                     model=model_name,  # 选择的模型（qwen-flash/max）
#                     openai_client=external_client,  # 绑定前面创建的通义千问客户端
#                 )
#             )
#         else:
#             agent = Agent(
#                 name="Assistant",
#                 instructions="",
#                 # 不关联MCP服务器（仅纯AI对话，不调用任何工具）
#                 model=OpenAIChatCompletionsModel(
#                     model=model_name,
#                     openai_client=external_client,
#                 )
#             )
#         # 运行Agent，以流式方式处理用户输入（prompt），并关联对话会话（session，openai-agent中定义的保存历史上下文）
#         result = Runner.run_streamed(agent, input=prompt, session=session)
#         # 异步遍历流式响应事件
#         async for event in result.stream_events():
#             # 筛选出"原始响应事件"，且数据是文本片段事件（ResponseTextDeltaEvent）
#             if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
#                 yield event.data.delta  # 生成器返回当前文本片段（实现流式输出）
#
# # 只有当用户输入的API Token长度大于1（已配置有效Token）时，才允许发送聊天请求
# if len(key) > 1:
#     # 监听用户的聊天输入框（st.chat_input()创建底部输入框，用户输入后返回内容）
#     if prompt := st.chat_input():
#         # 将用户输入添加到会话状态的消息列表中
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         # 渲染用户输入的消息
#         with st.chat_message("user"):
#             st.markdown(prompt)
#
#         # 渲染助手的响应消息框
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()  # 创建空占位符，用于动态更新响应内容
#             full_response = ""  # 存储完整的响应文本
#
#             with st.spinner("请求中..."):  # 显示加载中提示
#                 try:
#                     # 调用get_model_response获取流式响应生成器
#                     response_generator = get_model_response(prompt, model_name, use_tool)
#
#                     # 定义内部异步函数：遍历流式生成器，累积文本并更新界面
#                     async def stream_and_accumulate(generator):
#                         accumulated_text = ""
#                         async for chunk in generator:   # 异步遍历每个文本片段
#                             accumulated_text += chunk  # 累积文本
#                             # 在占位符中显示已累积的文本，并添加"▌"光标效果
#                             message_placeholder.markdown(accumulated_text + "▌")
#                         return accumulated_text  # 返回完整文本
#
#                     # 运行异步函数，获取完整响应
#                     full_response = asyncio.run(stream_and_accumulate(response_generator))
#                     # 响应完成后，移除光标，显示完整文本
#                     message_placeholder.markdown(full_response)
#
#                 except Exception as e:
#                     error_message = f"发生错误: {e}"
#                     message_placeholder.error(error_message)
#                     full_response = error_message
#                     print(f"Error during streaming: {e}")
#
#             # 将完整的助手响应添加到会话状态的消息列表中（保存历史）
#             st.session_state.messages.append({"role": "assistant", "content": full_response})

