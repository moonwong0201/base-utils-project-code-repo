"""
用 SQLAlchemy ORM 操作 SQLite 数据库，定义了知识库和文档的表结构
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
# - create_engine：创建数据库连接引擎（负责和数据库建立连接）
# - Column：定义表中的列（字段）
# - Integer/String/DateTime：数据类型（整数/字符串/日期时间）
# - ForeignKey：外键（用于表之间的关联）
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
# - declarative_base：创建模型基类（所有表模型都要继承它）
# - relationship：定义表之间的关系（比如“一个知识库包含多个文档”）
# - sessionmaker：创建会话工厂（用于操作数据库的“会话”对象）
from datetime import datetime

import yaml  # type: ignore

"""
主要负责配置数据库连接和定义数据库表结构，是整个 RAG 系统存储数据的基础。
它使用了 SQLAlchemy 库（Python 的 ORM 工具），让我们可以用 Python 类来操作数据库，而不用直接写 SQL 语句
"""

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)  # 把配置文件内容转换成Python字典

# 从配置中提取数据库相关设置
db_config = config['database']
db_type = db_config['engine']

# 数据库连接引擎创建，建立和数据库的连接
if db_type == "sqlite":
    # SQLite 使用文件路径
    db_path = db_config.get('path', 'rag.db')
    # 创建连接引擎：SQLite用文件路径标识数据库
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
else:
    # MySQL 或其他数据库使用 host, port, username, password
    host = db_config.get('host', 'localhost')   # 数据库服务器地址，默认本地
    port = db_config.get('port', 3306)    # 端口号，MySQL默认3306
    username = db_config.get('username', 'user')   # 登录用户名
    password = db_config.get('password', 'password')   # 登录密码
    database = db_config.get('database', 'mydb')   # 要连接的数据库名

    engine = create_engine(
        f"{db_type}://{username}:{password}@{host}:{port}/{database}",
        echo=True
    )

# 创建 Base 类
Base = declarative_base()


# ORM
# 定义知识库表（存储知识库的基本信息）
class KnowledgeDatabase(Base):
    __tablename__ = 'knowledge_database'  # 数据库表名

    # 字段定义（对应表中的列）
    knowledge_id = Column(
        Integer,            # 数据类型：整数
        primary_key=True,   # 主键（唯一标识一条记录，类似身份证号）
        autoincrement=True  # 自动递增（新增记录时不用手动指定，数据库自动生成）
    )
    title = Column(String)     # 知识库标题
    category = Column(String)  # 知识库分类
    create_dt = Column(
        DateTime,   # 数据类型：日期时间
        default=datetime.utcnow   # 默认值：创建记录时的UTC时间
    )
    update_dt = Column(
        DateTime,
        default=datetime.utcnow,   # 默认值：创建时的时间
        onupdate=datetime.utcnow   # 更新时自动刷新为当前时间
    )  # 更新时间

    # 与 KnowledgeDocument 表的关系：一个知识库包含多个文档（一对多关系）
    documents = relationship(
        "KnowledgeDocument",  # 关联的表模型
        back_populates="knowledge"     # 对应文档表中的'knowledge'字段
    )

    # 自定义打印格式（方便调试时查看对象信息）
    def __str__(self):
        return (f"KnowledgeDatabase(knowledge_id={self.knowledge_id}, "
                f"title='{self.title}', category='{self.category}', "
                f"author_id={self.author_id}, create_dt={self.create_dt}, "
                f"update_dt={self.update_dt})")


# 定义 knowledge_document 表  文档表
class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_document'   # 表名

    document_id = Column(Integer, primary_key=True, autoincrement=True)  # 文档主键，自动递增
    title = Column(String)     # 文档标题
    category = Column(String)  # 文档分类（如"PDF文档"）
    knowledge_id = Column(
        Integer,
        ForeignKey('knowledge_database.knowledge_id')  # 知识库主键（外键） 关联知识库表的knowledge_id
    )
    file_path = Column(String)  # 文档在服务器上的存储路径
    file_type = Column(String)  # 数据类型
    create_dt = Column(DateTime, default=datetime.utcnow)  # 创建时间
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 更新时间

    # 与 KnowledgeDatabase 表的关系：一个文档属于一个知识库（多对一关系）
    knowledge = relationship(
        "KnowledgeDatabase",
        back_populates="documents"  # 对应知识库表中的'documents'字段
    )

    # 解析状态
    status = Column(String, default="pending")  # pending/parsing/completed/failed
    error_msg = Column(String, default="")      # 失败原因



# 创建所有表（如果表不存在）
# 解释：Base.metadata包含了所有继承自Base的表结构，create_all会根据这些结构在数据库中创建表
Base.metadata.create_all(engine)

# 创建会话工厂（用于创建数据库会话）
# 解释：Session是一个类，每次调用Session()会创建一个“会话”对象，通过这个对象执行增删改查
Session = sessionmaker(bind=engine)
