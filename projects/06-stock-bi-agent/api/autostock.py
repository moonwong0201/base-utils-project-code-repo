"""
https://www.autostock.cn/#/trade/stock
https://s.apifox.cn/c3278b4f-5629-4732-858c-36758ff5d083/api-147275957
"""

import requests  # type: ignore
# 用于向 HTTP/HTTPS 服务器发送请求（比如 GET, POST）并获取响应。在这个文件里，它被用来调用 autostock.cn 的 API

from typing import Annotated  # 常用来结合 Path, Query, Body 等对象为接口参数添加描述和验证规则
from typing import Optional, Dict
import traceback
from fastapi import FastAPI, APIRouter, Query  # type: ignore
# APIRouter: 用于将不同功能的路由（接口）组织成独立的模块
from models.data_models import BasicResponse
from config import STOCK_TOKEN as TOKEN

# """
# 创建一个 FastAPI 服务，该服务封装了对 autostock.cn 这个第三方股票数据 API 的调用，
# 为前端或其他服务提供统一、便捷的股票数据查询接口
# """

app = FastAPI(
    name="Stock api Server",
    instructions="""This server provides stock basic tools.""",
)


# path get_stock_code http服务的路径
# operation_id 给这个接口操作指定一个唯一的 ID  mcp服务的名字
@app.get("/get_stock_code", operation_id="get_stock_code")
async def get_all_stock_code(
        keyWord: Annotated[Optional[str], Query(description="支持代码和名称模糊查询")] = None
) -> Dict:
    """所有股票，支持代码和名称模糊查询"""
    # 拼接出要请求的第三方 API 的完整 URL
    # "https://api.autostock.cn/v1/stock/all": 这是 autostock.cn 提供的 “获取所有股票” 接口的基础 URL。
    # "?": URL 中查询参数的开始标志。
    # "token=" + TOKEN: 将我们的身份验证令牌作为查询参数附加到 URL 末尾
    url = "https://api.autostock.cn/v1/stock/all" + "?token=" + TOKEN

    # url += "&keyWord=" + keyword: 在已有的 URL 后面继续拼接。
    # &: 用于分隔多个查询参数。
    # keyWord: 这是 autostock.cn API 要求的、用于模糊查询的参数名（注意大小写）。
    # keyword: 客户端传入的查询关键词
    if keyWord:
        url += "&keyWord=" + keyWord

    # payload: 通常用于存放 POST 请求的请求体数据。因为这个接口是 GET 请求，所以 payload 为空。
    # headers: 用于存放 HTTP 请求头信息。这里也为空，因为 autostock.cn 的这个 API 可能不需要额外的请求头。
    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        # "GET": 指定请求方法为 GET。
        # url: 目标 URL。
        # headers=headers: 传入请求头。
        # data=payload: 传入请求体数据（对于 GET 请求，这个参数会被忽略）。
        # timeout=10: 设置超时时间为 10 秒。如果在 10 秒内没有收到服务器的响应，requests 会抛出一个超时异常。
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)

        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


@app.get("/get_index_code", operation_id="get_index_code")
async def get_all_index_code():
    """所有指数，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/index/all" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


@app.get("/get_industry_code", operation_id="get_industry_code")
async def get_stock_industry_code():
    """获取板块数据"""
    url = "https://api.autostock.cn/v1/stock/industry/rank" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


@app.get("/get_board_info", operation_id="get_board_info")
async def get_stock_board_info():
    """获取大盘数据"""
    url = "https://api.autostock.cn/v1/stock/board" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


@app.get("/get_stock_rank", operation_id="get_stock_rank")
async def get_stock_rank(
        node: Annotated[str, Query(description="股票市场/板块代码: a,ash,asz,bsh,bsz")],
        industryCode: Annotated[Optional[str], Query(description="行业代码")] = None,
        pageIndex: Annotated[int, Query(description="页码")] = 1,
        pageSize: Annotated[int, Query(description="每页大小")] = 100,
        sort: Annotated[str, Query(description="排序字段")] = "price",
        asc: Annotated[int, Query(description="0=降序,1=升序")] = 0
) -> Dict:
    """股票价格排行"""
    url = "https://api.autostock.cn/v1/stock/rank" + "?token=" + TOKEN
    headers = {}  # type: ignore

    try:
        payload = {
            "node": node,
            "industryCode": industryCode,
            "pageIndex": pageIndex,
            "pageSize": pageSize,
            "sort": sort,
            "asc": asc
        }
        response = requests.post(url, json=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}


@app.get("/get_month_line", operation_id="get_month_line")
async def get_stock_month_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], Query(description="开始时间(非必填)")] = None,
        endDate: Annotated[Optional[str], Query(description="结束时间(非必填)")] = None,
        type: Annotated[int, Query(description="0不复权,1前复权,2后复权")] = 0
) -> Dict:
    """月k"""
    url = "https://api.autostock.cn/v1/stock/kline/month" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


@app.get("/get_week_line", operation_id="get_week_line")
async def get_stock_week_kline(
        code: Annotated[str, Query(description="股票代码")],
        startDate: Annotated[Optional[str], Query(description="开始时间(非必填)")] = None,
        endDate: Annotated[Optional[str], Query(description="结束时间(非必填)")] = None,
        type: Annotated[int, Query(description="0不复权, 1前复权, 2后复权")] = 0
):
    """周k"""
    url = "https://api.autostock.cn/v1/stock/kline/week" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


@app.get("/get_day_line", operation_id="get_day_line")
async def get_stock_day_kline(
        code: Annotated[str, Query(description="股票代码")],
        startDate: Annotated[Optional[str], Query(description="开始时间(非必填)")] = None,
        endDate: Annotated[Optional[str], Query(description="结束时间(非必填)")] = None,
        type: Annotated[int, Query(description="0不复权, 1前复权, 2后复权")] = 0
) -> Dict:
    """日k"""
    url = "https://api.autostock.cn/v1/stock/kline/day" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


@app.get("/get_stock_info", operation_id="get_stock_info")
async def get_stock_info(code: Annotated[str, Query(description="股票代码")]) -> Dict:
    """股票基础信息"""
    url = "https://api.autostock.cn/v1/stock" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


@app.get("/get_stock_minute_data", operation_id="get_stock_minute_data")
async def get_stock_minute_data(code: str):
    """分时信息"""
    url = "https://api.autostock.cn/v1/stock/min" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

