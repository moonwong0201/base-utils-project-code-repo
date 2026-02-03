import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Saying-MCP-Server",
    instructions="""This server contains some api of saying.""",
)


@mcp.tool
def get_today_familous_saying():
    # 功能说明：获取随机名言或“一言”
    """Retrieves a random famous saying or 'hitokoto' quote using the external API.[type: saying]"""
    try:
        # 调用一言API，解析返回的JSON数据中的"hitokoto"字段（名言内容）
        return requests.get(f"https://whyta.cn/api/yiyan?key={TOKEN}").json()["hitokoto"]
    except:
        return []


@mcp.tool
def get_today_motivation_saying():
    # 功能说明：获取励志语录或激励性名言
    """Retrieves a motivation saying or inspirational quote from the API.[type: saying]"""
    try:
        return requests.get(f"https://whyta.cn/api/tx/lzmy?key={TOKEN}").json()["result"]
    except:
        return []


@mcp.tool
def get_today_working_saying():
    # 功能说明：获取职场相关语录或心灵鸡汤内容
    """Retrieves a quote related to work or chicken soup for the soul (心灵鸡汤) content.[type: saying]"""
    try:
        return requests.get(f"https://whyta.cn/api/tx/lzmy?key={TOKEN}").json()["result"]["content"]
    except:
        return []
