import traceback
from typing import Optional

from fastapi import FastAPI, APIRouter, HTTPException  # type: ignore
import services.stock as service_stock
from pydantic import BaseModel
from models.data_models import BasicResponse, StockFavInfo

router = APIRouter(prefix="/v1/stock", tags=["stocks"])

"""
定义一个 FastAPI 路由模块，专门处理与 “用户自选股票” 相关的 Web API 接口。
它提供了查询、添加、删除和清空用户自选股票列表的功能
"""


class StockOperateRequest(BaseModel):
    user_name: str
    stock_code: Optional[str] = None


# 查询指定用户的所有自选股票列表
@router.post("/list_fav_stock", response_model=BasicResponse[list[StockFavInfo]])
async def get_user_all_stock(request: StockOperateRequest):
    try:
        stock_list = await service_stock.get_user_all_stock(request.user_name)
        if not stock_list:
            return BasicResponse(code=200, message="用户暂无自选股票", data=stock_list)
        # data: 调用服务层函数获取指定用户的所有自选股票
        return BasicResponse(code=200, message="获取用户所有股票成功", data=stock_list)
    except Exception as e:
        error_msg = f"获取用户自选股票失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=[])


# 从指定用户的自选股票列表中删除某一只股票
@router.post("/del_fav_stock", response_model=BasicResponse[bool])
async def delete_user_stock(request: StockOperateRequest):
    if not request.stock_code:
        return BasicResponse(code=400, message="股票代码不能为空", data=False)

    try:
        delete_result = await service_stock.delete_user_stock(request.user_name, request.stock_code)
        if delete_result:
            return BasicResponse(code=200, message="删除成功", data=delete_result)
        else:
            return BasicResponse(code=404, message="用户不存在或未收藏该股票", data=False)
    except Exception as e:
        error_msg = f"删除自选股票失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=[])


# 向指定用户的自选股票列表中添加某一只股票
@router.post("/add_fav_stock", response_model=BasicResponse[bool])
async def add_user_stock(request: StockOperateRequest):
    if not request.stock_code:
        return BasicResponse(code=400, message="股票代码不能为空", data=False)

    try:
        add_result = await service_stock.add_user_stock(request.user_name, request.stock_code)
        if add_result:
            return BasicResponse(code=200, message="添加成功", data=add_result)
        else:
            return BasicResponse(code=400, message="添加失败（用户不存在/股票代码无效/已收藏）", data=False)

    except Exception as e:
        error_msg = f"添加自选股票失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=False)


# 删除指定用户的所有自选股票
@router.post("/clear_fav_stock", response_model=BasicResponse[bool])
async def clear_user_stock(request: StockOperateRequest):
    try:
        clear_result = await service_stock.clear_user_stock(request.user_name)
        if clear_result:
            return BasicResponse(code=200, message="删除成功", data=clear_result)
        else:
            return BasicResponse(code=404, message="用户不存在或暂无自选股票", data=False)
    except Exception as e:
        error_msg = f"清空自选股票失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=[])


