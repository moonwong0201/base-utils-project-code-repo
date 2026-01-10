import os
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

load_dotenv()

def call_large_model(messages):
    """
    封装大模型API调用函数，增加异常处理
    :param messages: 大模型请求消息列表
    :return: 大模型响应结果
    """
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
        )
        completion = client.chat.completions.create(
            model=os.getenv("DASHSCOPE_MODEL"),
            messages=messages,
        )
        return completion
    except AuthenticationError:
        print("错误：API Key认证失败，请检查配置")
        return None
    except RateLimitError:
        print("错误：API调用频率超限，请稍后再试")
        return None
    except APIError as e:
        print(f"错误：大模型API调用失败，详情：{e}")
        return None


# 测试调用
if __name__ == "__main__":
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
        {"role": "user", "content": "帮我推导最小二乘算法？"},
    ]
    result = call_large_model(test_messages)
    if result:
        print(result.model_dump_json(indent=2))
