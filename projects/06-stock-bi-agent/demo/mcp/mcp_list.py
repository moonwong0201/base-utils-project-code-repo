import streamlit as st
import asyncio
import traceback
from fastmcp import Client
# 假设 Tool 在 fastmcp.tools 中，如用户代码所示
from fastmcp.tools import Tool
from typing import List
import pandas as pd

# """
# FastMCP 工具的 “信息展示面板”，核心作用是「可视化呈现所有已注册的 MCP 工具元信息」—— 不执行工具调用，
# 仅展示工具名称、功能描述、输入参数（类型 / 必填 / 默认值 / 描述），
# 是开发者 / 用户快速了解系统 “工具能力清单” 的入口，也是 MCP 工具的 “说明书页面”。
# """


# 辅助函数：判断工具类别 (已移除，根据用户要求)

# 确保 Streamlit 应用中使用 asyncio
# st.cache_data 缓存函数结果，避免每次 Streamlit 刷新都重新连接和获取数据
@st.cache_data(show_spinner="正在连接 FastMCP 服务器并获取工具列表...")
def load_mcp_tools(url: str) -> tuple[bool, List[Tool]]:
    """
    同步函数中运行异步客户端逻辑
    """

    async def get_data():
        client = Client(url)  # 连接 8900 端口的 MCP 服务器
        try:
            async with client:
                ping_result = await client.ping()  # 检测服务器连通性
                tools_list = await client.list_tools()  # 获取所有工具的完整元信息
                return ping_result, tools_list
        except Exception as e:
            # 捕获连接失败、超时等异常
            st.error(f"连接 FastMCP 服务器失败或发生错误: {e}")
            traceback.print_exc()
            return False, []

    # 运行异步函数并返回结果
    return asyncio.run(get_data())


def display_tool_info(tool: Tool):
    """
    以折叠框形式展示单个工具的详细输入参数
    """
    # 提取描述，只取到 **Responses:** 之前的部分作为摘要
    description_summary = tool.description.split('**Responses:**')[0].strip()

    # 移除了 get_tool_category 的调用，只显示工具名称
    with st.expander(f"**🔧 {tool.name}**"):
        st.markdown(f"**功能描述:**\n\n{description_summary}")

        # 提取并展示 Query Parameters
        if tool.inputSchema and 'properties' in tool.inputSchema:
            st.markdown("---")
            st.subheader("输入参数 (Query Parameters)")

            params = tool.inputSchema['properties']
            required = tool.inputSchema.get('required', [])

            # 整理参数信息为表格数据
            param_data = []
            for name, prop in params.items():
                is_required = name in required
                type_str = prop.get('type', 'Any')
                default_val = prop.get('default', '无')

                # 提取描述，如果存在的话
                param_desc = prop.get('description', '无描述')

                param_data.append({
                    "参数名": name,
                    "类型": type_str,
                    "必填": "✅" if is_required else "❌",
                    "默认值": default_val,
                    "描述": param_desc
                })

            if param_data:
                # 使用 DataFrame 创建表格
                st.dataframe(pd.DataFrame(param_data), hide_index=True)
            else:
                st.info("该工具没有输入参数。")


# --- Streamlit 主应用逻辑 ---
def main():
    MCP_SERVER_URL = "http://127.0.0.1:8900/sse"

    # 1. 服务器连接状态展示（顶部固定）
    status_container = st.container()
    with status_container:
        st.info(f"正在尝试连接服务端: `{MCP_SERVER_URL}`")

    # 调用函数加载数据
    ping_status, tools = load_mcp_tools(MCP_SERVER_URL)

    with status_container:
        if ping_status:
            st.success("✅ **客户端 Ping 成功!** 服务器连接状态良好。")
        else:
            st.error("❌ **客户端 Ping 失败!** 请检查服务器是否运行。")

    # 2. 工具总览表格（快速浏览所有工具）
    if tools:
        # 准备用于主列表的数据
        tool_list_data = []
        for tool in tools:
            tool_list_data.append({
                "工具名称": tool.name,
                # 移除了 "类别" 字段
                "功能摘要": tool.description.split('**Responses:**')[0].strip().split('\n')[0]  # 取第一行作为摘要
            })

        # 3. 工具详细信息（折叠框列表）
        st.subheader("工具总览")
        st.dataframe(pd.DataFrame(tool_list_data), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("工具详细信息 (展开查看参数)")

        # 循环展示每个工具的详细信息
        for tool in tools:
            display_tool_info(tool)


    else:
        st.warning("未能获取到任何工具信息。请检查上面的错误信息。")


print("2222")
main()

# 维度	    mcp_list.py	                mcp_debug.py	                chat.py
# 核心用途	展示工具元信息（说明书）	    调试工具（执行调用 + 看结果）	    业务使用（聊天 + 间接调用工具）
# 核心功能	连接服务器→展示工具列表 / 参数	连接服务器→选工具→填参数→执行调用	连接服务器→选工具→聊天→间接调用工具
# 目标用户	开发者 / 用户（了解工具能力）	开发者（调试工具）	                普通用户（使用功能）
# 交互方式	只读（无操作按钮）	            表单输入 + 调用按钮	            聊天输入 + 工具勾选
# 结果输出	无（仅展示信息）	            工具原始返回结果	                聊天回复 + K 线可视化

