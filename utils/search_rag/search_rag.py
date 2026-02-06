"""
这段代码实现了一个多智能体协作的问答系统，核心流程是：
用户提问 → 思考代理分解为子问题 → 搜索代理并行搜索子问题 → 总结代理整合结果 → 输出最终答案
"""

import asyncio
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["OPENAI_BASE_URL"] = "OPENAI_BASE_URL"

from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, ItemHelpers
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

import requests

# URL 解析 / 编码库（用于处理搜索关键词中的特殊字符，比如空格、中文）。
import urllib.parse


@function_tool  # 装饰器，将普通函数 search_jina 转换为 AI 代理可调用的工具
def search_jina(query: str) -> str:
    """通过jina进行谷歌搜索"""
    try:
        # 对查询字符串进行 URL 编码，确保特殊字符不会破坏 URL
        query = urllib.parse.quote(query)
        # 构建 Jina 搜索 API 的 URL，包含编码后的查询字符串和语言参数（中文）
        url = f"https://s.jina.ai/?q={query}&hl=zh-cn"
        headers = {  # 定义 HTTP 请求头
            "Accept": "application/json",  # 表示期望接收 JSON 格式的响应
            "Authorization": "替换成你的认证令牌",  # Jina API 的认证令牌
            "X-Respond-With": "no-content"  # Jina 的自定义请求头，通常用于简化返回结果格式（只返回核心内容，不返回冗余信息）
        }
        # 发送 GET 请求到 Jina 搜索 API
        response = requests.get(url, headers=headers)
        response.raise_for_status()  
        return response.text[:100] if response.text else "未获取到搜索结果"
    except requests.exceptions.Timeout:
        return "搜索超时，请重试"
    except requests.exceptions.HTTPError as e:
        return f"搜索接口错误：{e.response.status_code}，{e}"
    except Exception as e:
        return f"搜索失败：{str(e)}"  # 保留错误信息，方便调试


async def main(question: str):
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    thinking_agent = Agent(
        name="Thinking agent",
        instructions="你是一个擅长对问题进行分解的专家，为了搜索得到全面的答案，请对输入的提问拆分为不同的子问题，每行一个",
        model=OpenAIChatCompletionsModel(
            model="qwen-max",
            openai_client=external_client,  # 使用之前创建的异步 OpenAI 客户端
        ),
        # 模型的配置，parallel_tool_calls=False 表示不允许并行调用工具（因为这个代理不需要调用工具）
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    # 搜索代理
    search_agent = Agent(
        name="Search agent",
        instructions="你是一个搜索引擎，请使用jina进行谷歌搜索，并总结搜索结果",
        tools=[search_jina],  # 绑定之前定义的search_jina工具
        model=OpenAIChatCompletionsModel(
            model="qwen-max",
            openai_client=external_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    # 总结代理
    summary_agent = Agent(
        name="Summary agent",
        instructions="你是一个总结专家，结合用户的提问，总结搜索结果",
        model=OpenAIChatCompletionsModel(
            model="qwen-max",
            openai_client=external_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    sub_question = await Runner.run(
        thinking_agent,
        question,
    )
    sub_question = [x for x in sub_question.final_output.split("\n") if len(x) > 2]
    # 将代理的输出按换行符分割成列表。
    # 使用列表推导式过滤掉长度小于等于 2 的行（可能是空行或无效行）。
    # 将过滤后的子问题列表重新赋值给 sub_question

    print("分解后的问题:", sub_question)

    search_tasks = []
    for question in sub_question[:3]:
        if question.strip():
            # 只是创建任务，并没有执行
            result = Runner.run(
                search_agent,
                question
            )
            search_tasks.append(result)

    # 真正触发所有任务并发执行
    search_results = await asyncio.gather(*search_tasks)

    # 使用 ItemHelpers.text_message_outputs() 函数从每个搜索代理的结果中提取文本消息。
    outputs = [ItemHelpers.text_message_outputs(res.new_items) for res in search_results]
    merge_result = "\n\n".join(outputs)

    final_result = await Runner.run(
        summary_agent,
        f"原始提问：{question}\n 搜索结果:{merge_result}",
    )
    print(final_result.final_output)

if __name__ == "__main__":
     asyncio.run(main("如何学习机器学习？"))

