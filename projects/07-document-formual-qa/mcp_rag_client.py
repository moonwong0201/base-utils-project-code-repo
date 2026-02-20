"""
这段代码是「RAG 匹配 + LLM 工具调用 + MCP 服务执行」的客户端逻辑，核心流程是：
用户提问 → TF-IDF 计算相似度 → 匹配最相关的 MCP 工具 → 调用 Qwen-max 模型解析参数并调用工具 → 返回计算结果 + 自然语言回答
"""

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

import asyncio
from fastmcp import Client
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量器，计算文本相似度
import jieba
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# 导入agents相关模块：封装LLM调用、MCP服务对接、工具调用逻辑
from agents.mcp.server import MCPServerSse
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, ModelSettings
from agents.mcp import ToolFilterStatic
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

app = FastAPI(title="公式问答API")

# MCP客户端（连接本地8900端口的MCP服务）
client = Client("http://localhost:8900/sse")

class QAQuery(BaseModel):
    question: str

# 连接 MCP 服务，获取所有已封装的计算工具的名称和描述
async def list_tools():
    async with client:
        # 调用MCP服务的list_tools接口，获取所有已注册的工具
        result = await client.list_tools()
        # 提取工具名和工具描述（用于后续相似度匹配）
        names = [tool.name for tool in result]  # 待选工具的名字
        descs = [tool.description for tool in result]  # 待选工具的描述
        return names, descs


async def rag_tool_call(user_question: str):
    # 获取所有MCP工具的名称和描述
    function_names, function_descriptions = await list_tools()  # 列举所有的待选工具
    print("ALL tools")
    print(function_names)
    # 初始化TF-IDF向量器，处理中文文本
    tfidf = TfidfVectorizer()
    # 对工具描述进行中文分词 + TF-IDF向量化
    function_descriptions_tfidf = tfidf.fit_transform(
        [" ".join(jieba.lcut(x)) for x in function_descriptions]
    ).toarray()

    # 用户提问
    # 对用户问题进行同样的分词 + TF-IDF向量化（复用工具描述的TF-IDF词典）
    user_question_tfidf = tfidf.transform(
        [" ".join(jieba.lcut(user_question))]
    ).toarray()
    # 计算用户问题与每个工具描述的余弦相似度
    similarity = np.dot(user_question_tfidf, function_descriptions_tfidf.T)[0]
    # 按相似度从高到低排序，取前5个工具（候选工具列表）
    top5_idx = np.argsort(similarity)[::-1][:5]

    # 找到相似度最高的工具
    top5_tools = [function_names[i] for i in top5_idx]

    return top5_tools


async def llm_call_tool(user_question: str, top5_tools: list):
    # 设置工具过滤——只允许调用前5个匹配到的工具
    tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=top5_tools)
    # 初始化MCP服务连接（对接本地8900端口的MCP服务）
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,  # 应用工具过滤白名单
        client_session_timeout_seconds=20,
    )

    # 初始化异步OpenAI客户端（实际调用通义千问）
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # 创建Agent（智能体），负责协调LLM和MCP工具
    async with mcp_server:
        agent = Agent(
            name="Tool agent",
            # 给LLM的指令：复述问题→说明调用工具→输出结果→总结
            instructions="调用工具解决问题，首先复述用户问题，然后说明调用的工具，然后直接输出工具输出结果，最终总结。",
            mcp_servers=[mcp_server],  # 关联MCP服务
            model=OpenAIChatCompletionsModel(
                model="qwen-max",
                openai_client=external_client,
            ),
            model_settings=ModelSettings(parallel_tool_calls=False)  # 禁止并行调用工具，按顺序执行
        )

        # 流式运行Agent，处理用户问题
        result = Runner.run_streamed(agent, input=user_question, run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)))
        final_answer = ""
        # 遍历流式返回的事件，输出结果
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta') and event.data.delta:
                final_answer += event.data.delta
        return final_answer


@app.post("/qa")
async def qa_interface(query: QAQuery):
    try:
        top5_tools = await rag_tool_call(query.question)
        answer = await llm_call_tool(query.question, top5_tools)
        return {
            "code": 200,
            "matched_tools": top5_tools,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"后端执行错误：{str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
