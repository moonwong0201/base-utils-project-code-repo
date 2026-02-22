from datetime import datetime
from typing import Optional, List, TypeVar, Generic, Dict, Literal, Union, Any
from pydantic import BaseModel, Field, conint

# """
# 核心数据模型定义文件，基于 Pydantic 构建，统一了全项目的 “数据契约”—— 涵盖用户、响应、请求、会话、
# 股票等核心场景的数据结构，确保各模块（routers/services/agents）间数据传输的一致性和合法性
# """


# 定义用户的核心属性，用于 services/user_services.py 中用户信息的存储、查询和返回
class User(BaseModel):
    user_id: int    # 唯一用户 ID（整数型，必填）
    user_name: str  # 用户名（字符串型，必填）
    user_role: str  # 用户角色
    register_time: datetime  # 注册时间（datetime 类型，记录用户注册的具体时间点）
    status: bool = Field(True, description="用户状态，True=启用，False=禁用")    # 用户状态（布尔型，比如 True = 启用、False = 禁用）


# 泛型变量定义（支持BasicResponse泛型）
T = TypeVar('T')


# 所有 API 接口的统一响应格式，是 routers 层返回结果的标准模板
class BasicResponse(BaseModel, Generic[T]):
    code: int = Field(200, description="响应状态码，200=成功，400=参数错误，404=资源不存在，500=服务器错误")  # 响应状态码
    message: str = Field("操作成功", description="响应提示信息")  # 响应提示信息
    data: Optional[T] = Field(None, description="响应核心数据，泛型类型适配不同业务场景")  # 响应核心数据（可选值，支持列表 / 任意类型，无数据时可传 None）


# ---------routers/user.py 接口的请求体校验，确保前端传入的参数完整、合法---------
# 1. 用户登录请求
class RequestForUserLogin(BaseModel):
    user_name: str
    password: str


# 2. 用户注册请求
class RequestForUserRegister(BaseModel):
    user_name: str
    password: str
    user_role: str


# 3. 密码重置请求
class RequestForUserResetPassword(BaseModel):
    user_name: str
    password: str
    new_password: str


# 4. 用户信息修改请求
class RequestForUserChangeInfo(BaseModel):
    user_name: str
    user_role: Optional[str]
    status: Optional[bool]


# 用户在对话，传入的信息
# routers/chat.py 接口的请求体定义，接收用户的聊天输入及附加信息
class RequestForChat(BaseModel):
    content: str = Field(..., description="用户的提问")
    user_name: str = Field(..., description="用户名")
    session_id: Optional[str] = Field(None, description="对话session_id, 获取对话上下文")
    task: Optional[str] = Field(None, description="对话任务")
    tools: Optional[List[str]] = Field(None, description="可选的工具列表")

    # 后序可以持续增加，用户输入图、上传文件、链接、音频、视频，复杂的解析
    image_content: Optional[str] = Field(None)
    file_content: Optional[str] = Field(None)
    url_content: Optional[str] = Field(None)
    audio_content: Optional[str] = Field(None)
    video_content: Optional[str] = Field(None)

    # 后序可以持续增加，对话模型
    # 模式字段：控制智能体的工作模式（比如是否启用视觉分析、SQL 解析）
    vision_mode: Optional[bool] = Field(False)
    deepsearch_mode: Optional[bool] = Field(False)
    sql_interpreter: Optional[bool] = Field(False)
    code_interpreter: Optional[bool] = Field(False)


# routers/chat.py 接口的响应格式，返回 AI 对话结果及附加信息
# class ResponseForChat(BaseModel):
#     response_text: str  # 必选，AI 回复的核心文字内容
#     session_id: Optional[str] = Field(None)   # 可选，返回当前对话的会话 ID（用于多轮对话）
#     response_code: Optional[str] = Field(None)  # 可选，响应码
#     response_sql: Optional[str] = Field(None)  # 可选，若智能体调用了 SQL 工具，返回执行的 SQL 语句

class ResponseForChat(BaseModel):
    id: int
    create_time: datetime
    feedback: Optional[bool] = None
    feedback_time: Optional[datetime] = None
    role: str
    content: str
    generated_sql: Optional[str] = None
    generated_code: Optional[str] = None


# 定义用户自选股票的核心信息，用于 services/stock.py 中自选股的存储和查询
class StockFavInfo(BaseModel):
    stock_code: str  # 股票代码
    create_time: datetime


# 记录用户的对话会话信息，实现多轮对话上下文管理
class ChatSession(BaseModel):
    user_id: int  # 关联的用户 ID
    session_id: str  # 会话唯一标识
    title: str    # 会话标题
    start_time: datetime
    feedback: Optional[bool] = Field(None)  # 用户对会话的反馈
    feedback_time: Optional[datetime] = Field(None)
