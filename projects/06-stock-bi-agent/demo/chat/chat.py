import re
import time
import plotly.graph_objects as go

import requests
import streamlit as st
import asyncio
import traceback
import json
from fastmcp import Client
from fastmcp.tools import Tool
from typing import List, Any
import pandas as pd


# FastMCP 服务器地址
MCP_SERVER_URL = "http://127.0.0.1:8900/sse"

# """
# Streamlit 构建的聊天功能 Demo 前端，是项目 “智能聊天 + 股票工具调用” 的可视化演示入口，
# 核心逻辑是：通过 Streamlit 提供交互界面 → 调用后端 routers/chat.py/routers/stock.py 接口 → 实现 “流式聊天 + K 线可视化”，
# 完整串联了 “前端交互→后端接口→数据可视化” 的全流程
# """


# 连接后端 8900 端口的 MCP 服务器，获取所有已注册的工具列表（如 get_stock_day_kline），供前端用户勾选
# 缓存优化：@st.cache_data(ttl=60) 缓存工具列表 60 秒，避免每次刷新都请求服务器，提升性能；
# 容错设计：连接失败时打印异常并返回空列表，前端提示错误。
@st.cache_data(show_spinner="正在连接 FastMCP 服务器并获取工具列表...", ttl=60)
def load_mcp_tools(url: str) -> tuple[bool, List[Tool]]:
    """
    同步函数中运行异步客户端逻辑，获取所有可用工具。
    """

    async def get_data():
        client = Client(url)  # 连接 FastMCP 服务器（MCP工具服务）
        try:
            # 使用 async with 确保客户端连接正确管理
            async with client:
                ping_result = await client.ping()  # 检测服务器是否在线
                tools_list = await client.list_tools()  # 获取所有可用工具列表
                return ping_result, tools_list
        except Exception as e:
            st.error(f"连接 FastMCP 服务器失败或发生错误: {e}")
            traceback.print_exc()
            return False, []

    return asyncio.run(get_data())


# streamlit
# session_state 当前对话的缓存
# session_state.messages 此次对话的历史上下文

if st.session_state.get('logged', False):  # 登录校验：未登录则提示，已登录则显示用户名
    st.sidebar.markdown(f"用户名：{st.session_state['user_name']}")
else:
    st.info("请先登录再使用模型～")

# 初始化对话历史：首次进入/无session_id时，初始化系统提示词
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话 也 可以调用内部工具。"}
    ]
    st.session_state.messages_loaded = False  # 标记是否已加载历史

# ==== 新增：检测来源 ====
if st.session_state.get("from_history"):
    # 从 chat_list 点击历史进来的，保留 session_id，清除标记
    del st.session_state.from_history
elif st.session_state.get("session_id") and not st.session_state.get("loaded_session_id"):
    # 有 session_id 但没有 loaded_session_id，且不是从历史来的
    # 说明是从"通用对话"导航过来的残留状态，清空
    st.session_state.session_id = None
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话 也 可以调用内部工具。"}
    ]

current_session_id = st.session_state.get("session_id")
if current_session_id and st.session_state.get("loaded_session_id") != current_session_id:
    try:
        response = requests.post(
            f"http://127.0.0.1:8000/v1/chat/get",
            params={"session_id": st.session_state['session_id']}  # 用 params，不要空格
        )
        data = response.json()

        if data.get("code") == 200 and data.get("data"):
            # 清空默认消息，重新加载
            st.session_state.messages = [
                {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话 也 可以调用内部工具。"}
            ]
            for message in data["data"]:
                if message["role"] == "system":
                    continue
                st.session_state.messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

            # 标记当前已加载的session_id
            st.session_state.loaded_session_id = current_session_id

    except Exception as e:
        print(f"获取历史记录失败: {e}")

# 渲染历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 清空聊天历史函数
def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话 也 可以调用内部工具。"}
    ]
    st.session_state.session_id = None
    st.session_state.loaded_session_id = None  # 重置加载标记


# 展示 MCP 工具列表，供用户勾选本次对话可用的工具（如勾选 get_day_line）；
# 提供 “清空聊天” 按钮，调用 clear_chat_history 重置会话状态
with st.sidebar:
    if st.session_state.get('logged', False):
        ping_status, all_tools = load_mcp_tools(MCP_SERVER_URL)

        if not ping_status or not all_tools:
            st.error("未能加载工具。请检查服务器是否已在 8900 端口运行，并查看上方错误详情。")
            selected_tool_names = []
        else:
            # 将工具列表转换为 {name: Tool} 字典，方便查找
            tool_map = {tool.name: tool for tool in all_tools}
            tool_names = list(tool_map.keys())

            selected_tool_names = st.multiselect(
                "选择MCP工具:",
                options=tool_names,
            )

    st.button('清空当前聊天', on_click=clear_chat_history, use_container_width=True)


# 异步流式请求后端聊天接口
async def request_chat(content: str, user_name: str, session_id: str) -> str:
    url = "http://127.0.0.1:8000/v1/chat/"

    headers = {
        "accept": "text/event-stream",  # 修改为接受事件流
        "Content-Type": "application/json"
    }

    data = {
        "content": content.text if hasattr(content, 'text') else str(content),
        "user_name": user_name,
        "session_id": session_id,
        "stream": True,
        "tools": selected_tool_names  # 前端勾选的工具列表
    }

    if not session_id:
        del data["session_id"]

    try:
        # 流式请求：stream=True，迭代响应内容
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=30)
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):  # ← 用 iter_lines 代替 iter_content
            if line and line.startswith("data: "):
                # 去掉 "data: " 前缀
                chunk = line[6:]  # ← 关键修复
                if chunk:
                    yield chunk

    except requests.exceptions.Timeout:
        yield "错误：请求超时，请稍后重试"
    except requests.exceptions.ConnectionError:
        yield "错误：无法连接到服务器"
    except Exception as e:
        yield f"错误：{str(e)}"


# 用户首次发起聊天时，调用后端 /v1/chat/init 接口生成唯一 session_id，用于关联多轮对话
def request_session_id() -> str:
    url = "http://127.0.0.1:8000/v1/chat/init"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers)
    return response.json()["data"]["session_id"]


# 调用后端 /stock/get_day_line/get_week_line/get_month_line 接口，获取原始 K 线数据并清洗为 DataFrame
def fetch_k_line_data(
        endpoint: str,
        code: str,
        line_type: str,
        start_date: str,
        end_date: str,
        data_type: int = 0  # 假设 type=0 是默认的数据类型
):
    """
    通过调用后端 API 获取 K 线数据。
    """

    BASE_URL = "http://127.0.0.1:8000/stock/"
    url = f"{BASE_URL}{endpoint}"

    # 注意：您的 curl 示例中，日期参数被双引号包裹，但在 Python requests 中，
    # 传递日期字符串通常不需要额外的引号，后端应自行解析。
    params = {
        "code": code,
        "startDate": start_date,
        "endDate": end_date,
        "type": data_type,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("code") == 200 and data.get("data"):
            # 假设返回的数据结构是列表的列表：
            # [ ["日期", "昨收", "今开", "最高", "最低", "成交量"], ... ]

            # 数据清洗：转换为DataFrame，修正字段名和数据类型
            df = pd.DataFrame(data["data"])
            df = df.iloc[:, :6]
            df.columns = [
                "Date", "Close_Prev", "Open", "High", "Low", "Volume"
            ]

            # 转换为正确的数据类型
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ["Open", "High", "Low", "Close_Prev", "Volume"]:
                # 将数据类型转换为浮点数，并处理可能存在的错误值
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.rename(columns={'Close_Prev': 'Close'}, inplace=True)

            return df
        else:
            st.warning(f"API 返回成功，但未找到 {code} 的 K 线数据。")
            return None

    except requests.exceptions.ConnectionError:
        st.error(f"连接错误：无法连接到后端服务 ({BASE_URL})。请确保后端服务正在运行。")
        return None
    except Exception as e:
        st.error(f"获取 K 线数据时发生错误：{e}")
        traceback.print_exc()
        return None


def plot_candlestick(df: pd.DataFrame, code: str, line_type: str):
    """
    使用 Plotly 绘制交互式 K 线图。
    """

    # 确保数据按日期排序
    df = df.sort_values(by='Date')
    # 绘制蜡烛图（K线）
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K线'
    )])

    # 添加成交量 (Volume) 作为子图
    fig_volume = go.Figure(data=[go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='成交量'
    )])

    # 合并图表 (使用 make_subplots 可能会更好，但这里简化为两个独立的图)
    # 调整布局
    fig.update_layout(
        title=f"股票 K 线图 - {code} ({line_type})",
        xaxis_rangeslider_visible=False,  # 隐藏底部的时间轴滑动条
        xaxis=dict(title='日期'),
        yaxis=dict(title='价格'),
        hovermode="x unified",
        height=600  # 增加高度
    )

    # 绘制成交量图（如果需要合并子图，需要使用 plotly.subplots.make_subplots）
    # 在 Streamlit 中，通常将它们分开显示更简单
    st.plotly_chart(fig, use_container_width=True)

    fig_volume.update_layout(
        title="成交量 Volume",
        xaxis=dict(title='日期', showticklabels=True),
        yaxis=dict(title='成交量'),
        height=200
    )
    st.plotly_chart(fig_volume, use_container_width=True)


if prompt := st.chat_input(accept_file="multiple", file_type=["txt", "pdf", "jpg", "png", "jpeg", "doc", "docx"]):
    # 首次聊天生成session_id
    if "session_id" not in st.session_state.keys() or not st.session_state.session_id:
        st.session_state.session_id = request_session_id()

    if st.session_state.get('logged', False):
        # 记录用户输入到会话状态
        st.session_state.messages.append({"role": "user", "content": prompt.text})
        with st.chat_message("user"):  # 用户输入
            st.markdown(prompt.text)

        # 渲染AI回复（流式）
        with st.chat_message("assistant"):  # 大模型输出
            message_placeholder = st.empty()
            placeholder = st.empty()

            with st.spinner("请求中..."):
                async def stream_output():
                    accumulated_text = ""
                    # 调用流式请求函数
                    response_generator = request_chat(prompt, st.session_state['user_name'], st.session_state['session_id'])
                    async for data in response_generator:
                        accumulated_text += data
                        placeholder.markdown(accumulated_text + "▌")  # 后端不断sse输出内容，前端通过markdown渲染

                    return accumulated_text

                # 执行异步流式输出
                final_text = asyncio.run(stream_output())
                placeholder.markdown(final_text)  # 输出完成后移除光标

            # 记录AI回复到会话状态
            st.session_state.messages.append({"role": "assistant", "content": final_text})

            try:
                # 如果tool是如下的可视化的工具，则需要调用得到原始数据再进行绘图
                if "get_day_line" in final_text or "get_week_line" in final_text or "get_month_line" in final_text:

                    # 解析后端返回的工具调用结果（JSON片段）
                    function_json = re.search(r"```json\s*([\s\S]*?)\s*```", final_text, re.I).group(1).strip()
                    function_json = function_json.strip()
                    endpoint = function_json[:function_json.index(":")]  # 工具名字
                    argv = json.loads(function_json[function_json.index(":")+1:])  # 工具传入参数
                    stock_code = argv["code"]
                    start_date_str = argv["startDate"]
                    end_date_str = argv["endDate"]
                    line_type = argv["type"]
                    with st.spinner(f"正在加载 {stock_code} 数据 ({start_date_str} 至 {end_date_str})..."):
                        df_k_line = fetch_k_line_data(
                            endpoint=endpoint,
                            code=stock_code,
                            line_type=line_type,
                            start_date=start_date_str,
                            end_date=end_date_str
                        )
                        # 绘图
                        if df_k_line is not None and not df_k_line.empty:
                            st.success(f"成功加载 {len(df_k_line)} 条数据。")
                            plot_candlestick(df_k_line, stock_code, line_type)
                        else:
                            st.info("没有数据可以绘制 K 线图。请检查代码或日期范围。")
            except:
                traceback.print_exc()

# Demo 前端逻辑	    调用的后端接口 / 模块	                    核心交互内容
# 初始化 session_id	routers/chat.py → /v1/chat/init	        获取唯一对话标识
# 拉取历史对话	    routers/chat.py → /v1/chat/get	        按 session_id 恢复聊天上下文
# 流式聊天请求	    routers/chat.py → /v1/chat/	            传递用户输入、工具列表，接收 SSE 响应
# 获取 K 线数据	    routers/stock.py → /stock/get_*_line	传递股票代码、日期，获取 K 线原始数据
# 加载 MCP 工具列表	FastMCP 服务器（8900 端口）	            获取所有已注册的工具名
