import os
import logging

# 配置根日志级别为 INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from config import (
    OPENAI_API_KEY, OPENAI_VISION_MODEL, OPENAI_MODEL, OPENAI_BASE_URL
)

# """
# FastAPI 主服务启动文件，相当于整个系统的 “总控制台”—— 负责初始化环境、注册所有接口、启动服务，
# 是用户请求能到达后端的第一个关键节点。
#
# 它是服务启动的开关：运行这个文件，整个后端才会启动；
# 它是接口的总入口：所有外部请求都先到这里，再被转发到对应的业务模块；
# 它是全局配置的中心：统一管理日志、大模型密钥、服务端口等核心配置。
#
# 用户/前端/MCP → 访问 http://localhost:8000/* → main_server.py 的 app 实例 → 转发到对应路由/子应用
# """

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
os.environ["OPENAI_MODEL"] = OPENAI_MODEL
os.environ["OPENAI_VISION_MODEL"] = OPENAI_VISION_MODEL

import uvicorn
from fastapi import FastAPI  # type: ignore
from routers.user import router as user_routers
from routers.chat import router as chat_routers
from routers.data import router as data_routers
from routers.stock import router as stock_routers

from api.autostock import app as stock_app

app = FastAPI()


# 定义健康检查接口：用于服务监控
@app.get("/v1/healthy")
def read_healthy():
    pass  # 返回空响应，仅用于确认服务是否正常运行


# 把分散在 routers/ 目录下的所有业务接口（聊天、用户、数据、股票），统一挂载到主应用 app 上
# 比如：
# chat_routers 里的 /v1/chat/ 接口，注册后才能通过 http://localhost:8000/v1/chat/ 访问；
# 如果不注册，这些接口就 “隐身” 了，用户访问会返回 404。
app.include_router(user_routers)
app.include_router(chat_routers)
app.include_router(data_routers)
app.include_router(stock_routers)

# 挂载子应用：将独立的股票 API 应用挂载到 /stock 路径下
# 这样访问 /stock/* 路径的请求会被转发到 stock_app 处理
app.mount("/stock", stock_app)  # 底层stock api 接口

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn main_server:app
