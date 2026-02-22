from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, create_engine, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
# 字段类型（Column+Integer/String 等）：定义数据库表字段的类型和约束；
# DeclarativeBase：SQLAlchemy 2.0+ 推荐的 ORM 基类；
# sessionmaker：创建数据库会话（操作数据库的 “连接”）；
# Mapped/mapped_column：类型注解风格的字段定义（更简洁，支持类型提示）。

from config import DATABASE_URL

# """
# 数据库 ORM（对象关系映射）模型定义文件，基于 SQLAlchemy 构建，负责将 Python 类与数据库表进行映射，
# 定义了项目所有核心数据的存储结构（用户、数据文件、自选股、聊天会话 / 消息），是 services 层操作数据库的基础
# """

class Base(DeclarativeBase):
    pass


# 对应 services/user.py 中的用户注册、登录、信息修改等逻辑；
# 存储系统所有用户的基础信息，是其他表的 “主表”（通过 user.id 关联）。
class UserTable(Base):
    __tablename__ = 'user'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # 主键ID（自增）
    user_name: Mapped[str] = mapped_column(String)  # 用户名（唯一，用于登录）
    user_role: Mapped[str] = mapped_column(String)   # 用户角色（如"admin"/"user"）
    password: Mapped[str] = mapped_column(String)  # 密码
    register_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 注册时间
    status: Mapped[bool] = mapped_column(Boolean)  # 账号状态（True=启用，False=禁用）


# 关联业务：对应 routers/data.py 的文件上传、下载、删除等接口；
# 核心作用：记录用户上传的结构化数据文件信息，实现 “用户 - 文件” 的关联管理
class DataTable(Base):
    __tablename__ = 'data'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # 数据文件唯一ID
    path: Mapped[str] = mapped_column(String)  # 文件存储路径
    data_type: Mapped[str] = mapped_column(String)   # 文件类型
    create_user_id: Mapped[int] = mapped_column(Integer, ForeignKey('user.id'))  # 关联上传用户（外键→user表id）
    create_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 创建时间
    alter_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)   # 最后修改时间


# 关联业务：对应 routers/stock.py 的添加 / 删除 / 查询自选股接口；
# 关联模型：与 data_models.py 中的 StockFavInfo 对应（stock_id= stock_code，create_time 一致）；
# 核心作用：存储用户的自选股票列表，实现 “一人多股” 的关联。
class UserFavoriteStockTable(Base):
    __tablename__ = 'user_favorite_stock'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # 自增ID
    stock_id: Mapped[str] = mapped_column(String(20), nullable=False)  # 股票代码
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey('user.id'))  # 关联用户（外键→user表id）
    create_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 添加时间

    # 联合唯一约束：一个用户不能重复添加同一支股票
    __table_args__ = (
        UniqueConstraint('user_id', 'stock_id', name='_user_stock_uc'),
    )


# 核心作用：记录用户的每一次完整对话的 “整体信息”，是 ChatMessageTable 的 “主表”；
# 关联逻辑：一个用户（user_id）可以有多个会话（chat_session），一个会话对应多条消息（chat_message）。
class ChatSessionTable(Base):
    """表1，存储一次对话列表的基础信息，聊天会话模型：存储用户会话的元数据。"""
    __tablename__ = 'chat_session'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)  # 会话元数据ID
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey('user.id'), nullable=False)   # 关联用户
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)  # 会话唯一标识（与 `data_models.py` 的 `ChatSession.session_id` 一致）
    title: Mapped[str] = mapped_column(String(100))  # 会话标题
    start_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 会话开始时间
    feedback: Mapped[bool] = mapped_column(Boolean, nullable=True)  # 用户反馈
    feedback_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)


# 每个聊天记录的每次对话
# 存储会话中的每一条具体消息，配合 ChatSessionTable 实现 “多轮对话记忆”；
class ChatMessageTable(Base):
    """表2，存储一次对话的每一条记录，聊天消息模型：存储会话中的每一条消息。"""
    __tablename__ = 'chat_message'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)  # 单条消息ID
    chat_id: Mapped[int] = mapped_column(Integer, ForeignKey('chat_session.id'), nullable=False)  # 关联会话（外键→chat_session表id）
    role: Mapped[str] = mapped_column(String(10), nullable=True)  # 消息发送者角色
    content: Mapped[str] = mapped_column(Text, nullable=True)  # 消息内容（用户提问/AI回复文本）
    generated_sql: Mapped[str] = mapped_column(Text, nullable=True)  # 预留字段：AI生成的SQL语句
    generated_code: Mapped[str] = mapped_column(Text, nullable=True)  # 预留字段：AI生成的代码
    create_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    feedback: Mapped[bool] = mapped_column(Boolean, nullable=True)
    feedback_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)


database_url = DATABASE_URL
engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False}  # SQLite 特有配置：允许跨线程访问
)

Base.metadata.create_all(bind=engine)  # 自动创建所有表（表不存在时创建，已存在则不重复创建）

# 创建会话工厂：用于 services 层获取数据库连接会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
