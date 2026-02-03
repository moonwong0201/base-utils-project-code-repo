import requests
TOKEN = "738b541a5f7a"  # 定义API访问令牌（密钥），用于调用外部API时的身份验证

from fastmcp import FastMCP
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""This server contains some api of news.""",
)


# 用@mcp.tool装饰器定义一个工具函数，注册到MCP服务器中
@mcp.tool
def get_today_daily_news():
    # 函数文档字符串：说明功能是从外部API获取今日新闻简报列表
    """Retrieves a list of today's daily news bulletin items from the external API.[type: news]"""
    try:
        # 发送GET请求到指定API，携带TOKEN参数；解析返回的JSON数据，提取"result"中的"list"字段（新闻列表）
        return requests.get(f"https://whyta.cn/api/tx/bulletin?key={TOKEN}").json()["result"]["list"]
    except:
        return []


# 注册为MCP工具函数
@mcp.tool
def get_douyin_hot_news():
    # 功能说明：获取抖音热门话题/新闻列表
    """Retrieves a list of trending topics or hot news from Douyin (TikTok China) using the API.[type: news]"""
    try:
        # 调用抖音热点API，解析返回的JSON数据中的"result"->"list"字段
        return requests.get(f"https://whyta.cn/api/tx/douyinhot?key={TOKEN}").json()["result"]["list"]
    except:
        return []


@mcp.tool
def get_github_hot_news():
    # 功能说明：获取GitHub热门仓库/项目列表
    """Retrieves a list of trending repositories/projects on GitHub using the API.[type: news]"""
    try:
        # 调用GitHub热点API，解析返回的JSON数据中的"items"字段（热门项目列表）
        return requests.get(f"https://whyta.cn/api/github?key={TOKEN}").json()["items"]
    except:
        return []

@mcp.tool
def get_toutiao_hot_news():
    # 功能说明：获取今日头条热门新闻列表
    """Retrieves a list of hot news headlines from Toutiao (a Chinese news platform) using the API.[type: news]"""
    try:
        # 打印API请求地址（用于调试，确认请求URL是否正确）
        print(f"https://whyta.cn/api/tx/topnews?key={TOKEN}")
        # 调用今日头条API，解析返回的JSON数据中的"result"->"list"字段
        return requests.get(f"https://whyta.cn/api/tx/topnews?key={TOKEN}").json()["result"]["list"]
    except:
        import traceback
        traceback.print_exc()
        return []

@mcp.tool
def get_sports_news():
    # 功能说明：获取电竞或体育新闻列表
    """Retrieves a list of esports or general sports news items using the external API.[type: news]"""
    try:
        # 调用体育新闻API，解析返回的JSON数据中的"result"->"newslist"字段
        return requests.get(f"https://whyta.cn/api/tx/esports?key={TOKEN}").json()["result"]["newslist"]
    except:
        return []
