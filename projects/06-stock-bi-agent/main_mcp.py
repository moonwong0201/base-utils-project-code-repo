import asyncio
import traceback

import requests  # type: ignore
from fastmcp import FastMCP, Client

from api.autostock import app
from api.news import mcp as news_mcp
from api.saying import mcp as saying_mcp
from api.tool import mcp as tool_mcp

# """
# MCP 工具服务的「启动器 + 工具注册中心」，所有工具都要先注册到这里，才能被 Agent 调用
# 在 Agent 需要调用工具时起作用，且是 Agent 和工具 API 之间的唯一通信桥梁
# Agent 永远只和 MCP 通信，不会直接接触工具 API
#
# 维度	       main_server.py	       main_mcp.py
# 服务端口	   8000（对外，前端直接访问）  8900（对内，仅服务层访问）
# 核心作用	   提供前端可访问的业务接口	   提供 Agent 可调用的工具能力
# 通信对象	   前端（用户）	           服务层的 Agent
# 协议 / 方式  HTTP/RESTful（普通接口）  SSE（Server-Sent Events）
# 依赖关系	   不依赖 MCP 服务	       依赖工具 API（api/autostock.py 等）
# """

# 从 FastAPI 应用创建 FastMCP 实例
# FastMCP 是一个轻量级的微通信协议，这里将 FastAPI 应用与 MCP 服务绑定
# 初始形态：纯 FastAPI HTTP 接口（为了给前端 / 其他服务直接调用，比如main_server挂载的/stock/get_stock_code）；
# 注册方式：通过FastMCP.from_fastapi(app=app) 转换为 MCP 服务，实现 “一份代码，两用”；
mcp = FastMCP.from_fastapi(app=app)


# 定义异步初始化函数：注册其他 MCP 服务
async def setup():
    try:
        # 将 news_mcp、saying_mcp、tool_mcp 注册到主 MCP 服务中，这样 MCP 就能管理所有工具（股票、新闻、花语）
        # prefix="" 表示不添加额外的前缀，直接使用服务自身的路径
        await mcp.import_server(news_mcp, prefix="")
        await mcp.import_server(saying_mcp, prefix="")
        await mcp.import_server(tool_mcp, prefix="")
    except Exception as e:
        print(traceback.format_exc())
        raise


# 定义异步测试函数：验证服务注册是否成功
async def test_filtering():
    # 使用 Client 连接到本地 MCP 服务
    async with Client(mcp) as client:
        # 列出所有已注册的工具（服务）
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])
        print("Available tools:", [t for t in tools])


if __name__ == "__main__":
    # 执行初始化：注册其他服务
    asyncio.run(setup())
    # 执行测试：验证注册结果
    asyncio.run(test_filtering())
    # 启动 MCP 服务，使用 SSE (Server-Sent Events) 传输协议，监听 8900 端口
    mcp.run(transport="sse", port=8900)
