import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-078ae61448344f53b3cb03bcc85ff7cd"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
# 导入 OpenAI 流式响应的文本片段事件类型，用于筛选并处理 AI 返回的流式文本（逐字 / 逐句返回的内容）
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

"""
 AI+MCP 工具的端到端测试脚本，核心作用是跳过前端界面，
 直接验证 “AI 理解用户需求→调用 MCP 工具→获取数据→生成自然语言响应” 的完整流程，
 重点测试 AI 的工具调用能力和响应逻辑
"""


# 定义异步函数run：核心测试逻辑，创建AI代理并执行两次测试查询
async def run(mcp_server: MCPServer):
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    # 创建AI代理（Agent）：关联MCP服务，具备工具调用能力
    agent = Agent(
        name="Assistant",
        instructions="你是qwen，擅长回答各类问题。",
        mcp_servers=[mcp_server],  # 将 MCP 服务器客户端关联到 AI 代理
        model=OpenAIChatCompletionsModel(
            model="qwen-flash",
            openai_client=external_client,  # 绑定通义千问客户端
        )
    )

    message = "最近有什么新闻？"
    print(f"Running: {message}")

    """
    送入qwen-flash的是什么？
    [
        {
            "role": "system",
            "content": "你是qwen，擅长回答各类问题。"
        },
        {
            "role": "user",
            "content": "最近有什么新闻？get_today_daily_news调用结果"
        }
    ]
    """

    # 运行AI代理，以流式方式处理输入（message），获取响应结果
    result = Runner.run_streamed(agent, input=message)
    # 异步遍历流式响应事件
    async for event in result.stream_events():
        # 筛选出“原始响应事件”且数据为文本片段（排除其他类型事件）
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    message = "武汉最近的天气怎么样？"
    print(f"Running: {message}")

    # input：传给代理的核心输入内容
    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        # 筛选出“原始响应事件”且数据为文本片段（排除其他类型事件）
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


# 定义异步主函数：创建MCP客户端并调用run函数执行测试
async def main():
    # 异步创建MCP的SSE客户端，连接本地8900端口的MCP服务器
    async with MCPServerSse(
            name="SSE Python Server",  # MCP客户端名称
            params={
                "url": "http://localhost:8900/sse",  # MCP服务器的SSE接口地址
            },
    )as server:  # 上下文管理器自动管理MCP连接的建立与关闭
        await run(server)  # 传入MCP服务器实例，执行测试逻辑

if __name__ == "__main__":
    asyncio.run(main())
