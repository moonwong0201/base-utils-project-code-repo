"""
FastMCP 服务器的启动入口文件，负责整合项目中所有业务工具（news、saying、tool 模块）并启动服务器
"""
import asyncio
from fastmcp import FastMCP, Client

from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp

mcp = FastMCP(
    name="MCP-Server"
)


# 定义异步函数setup，用于初始化服务器（导入其他模块的MCP配置）
async def setup():
    # 导入news_mcp的配置到主mcp中，prefix=""表示不添加前缀（即使用原路径）
    await mcp.import_server(news_mcp, prefix="")
    # 导入saying_mcp的配置到主mcp中，不添加前缀
    await mcp.import_server(saying_mcp, prefix="")
    # 导入tool_mcp的配置到主mcp中，不添加前缀
    await mcp.import_server(tool_mcp, prefix="")


# 定义异步函数test_filtering，用于测试服务器（列出可用工具）
async def test_filtering():
    # 创建一个与主mcp绑定的客户端，使用async with确保客户端资源自动释放
    async with Client(mcp) as client:
        # 调用客户端方法获取所有可用工具列表
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    # 运行setup异步函数（执行服务器初始化，导入其他模块配置）
    asyncio.run(setup())
    # 运行test_filtering异步函数（执行工具列表测试）
    asyncio.run(test_filtering())
    # 启动MCP服务器，使用sse（Server-Sent Events）传输方式，监听8900端口
    mcp.run(transport="sse", port=8900)
