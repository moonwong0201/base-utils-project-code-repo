import asyncio
import logging
import traceback
import requests
from typing import Optional, List
from datetime import datetime

from api.autostock import get_stock_info
from models.data_models import StockFavInfo
from models.orm import UserFavoriteStockTable, SessionLocal, UserTable

# """
# 股票业务的核心服务层，专注处理「用户自选股的增删改查」，是 routers/stock.py 接口的直接支撑，
# 衔接了 “用户自选股接口” 和 “数据库 / 股票工具层”，逻辑简洁但覆盖了自选股管理的全核心场景
# """

logger = logging.getLogger(__name__)

# 给前端返回 “用户自选股列表”
async def get_user_all_stock(user_name: str) -> List[StockFavInfo]:  # StockFavInfo: 用户自选股票的核心信息
    with SessionLocal() as session:
        # 1. 查用户ID（关联UserTable）
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return []
        else:
            user_id = user_id[0]
        # 2. 查该用户的所有自选股记录
        user_stock_db_records = session.query(UserFavoriteStockTable).filter(UserFavoriteStockTable.user_id == user_id).all()
        # 3. 转换为data_models的StockFavInfo模型（适配前端数据格式）
        return [
            StockFavInfo(
                stock_code=user_stock_db_record.stock_id,   # ORM的stock_id → 传输模型的stock_code
                create_time=user_stock_db_record.create_time
            ) for user_stock_db_record in user_stock_db_records
        ]


# 用户在前端点击 “取消收藏” 某只股票时，删除数据库中对应的记录
async def delete_user_stock(user_name: str, stock_code: str) -> bool:
    with SessionLocal() as session:
        # 1. 查用户ID
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return False  # 用户不存在，默认返回“删除成功”（避免前端报错）
        else:
            user_id = user_id[0]
        # 2. 查该用户是否收藏了这只股票
        # 表示这个变量的类型要么是 UserFavoriteStockTable，要么是 None
        user_stock_db_record: UserFavoriteStockTable | None = session.query(UserFavoriteStockTable).filter(
            UserFavoriteStockTable.user_id == user_id, UserFavoriteStockTable.stock_id == stock_code).first()
        # 3. 存在则删除，提交变更
        if user_stock_db_record:
            session.delete(user_stock_db_record)
            session.commit()
            return True
        else:
            return False


# 核心作用：用户在前端点击 “收藏” 某只股票时，添加到数据库；
# 防重逻辑：先查是否已收藏，避免同一只股票被重复添加到自选股列表
async def add_user_stock(user_name: str, stock_code: str) -> bool:
    with SessionLocal() as session:
        # 1. 查用户ID
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return False
        else:
            user_id = user_id[0]

        try:
            # 调用api/autostock.py的get_stock_info，校验股票代码是否有效
            stock_info = await get_stock_info(stock_code)

            if not stock_info:  # 无返回结果 → 股票代码无效
                return False

            if stock_info.get("code") != 200:  # 业务状态码判断
                return False

            data = stock_info.get("data")
            if not data or len(data) == 0:  # data数组为空
                return False

        except Exception as e:
            # 捕获工具调用异常（如网络错误、接口超时），返回添加失败
            print(f"校验股票代码失败：{e}")
            print(traceback.format_exc())
            return False

        # 2. 校验：该用户是否已收藏这只股票（避免重复添加）
        user_stock_db_record: UserFavoriteStockTable | None = session.query(UserFavoriteStockTable).filter(
            UserFavoriteStockTable.user_id == user_id, UserFavoriteStockTable.stock_id == stock_code).first()
        # 3. 未收藏则添加，已收藏返回False（前端提示“已收藏”）
        if user_stock_db_record:
            return False
        else:
            user_stock_db_record = UserFavoriteStockTable(
                stock_id=stock_code,
                user_id=user_id,
                create_time=datetime.utcnow()
            )
            session.add(user_stock_db_record)
            session.commit()

            return True


# 核心作用：用户点击 “清空自选股” 时，批量删除该用户的所有自选股记录；
# 高效设计：用 delete() 批量删除，比循环删除单条记录效率更高
async def clear_user_stock(user_name: str) -> bool:
    with SessionLocal() as session:
        # 1. 查用户ID
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return False
        else:
            user_id = user_id[0]
        # 2. 批量删除该用户的所有自选股
        delete_count = session.query(UserFavoriteStockTable).filter(UserFavoriteStockTable.user_id == user_id).delete()
        session.commit()

        return delete_count > 0
