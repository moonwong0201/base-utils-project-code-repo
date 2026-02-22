import hashlib
import traceback
from typing import Optional, List
from datetime import datetime
from models.orm import UserTable, SessionLocal
from models.data_models import User

# """
# 用户管理的核心服务层，覆盖了用户 “注册 / 登录 / 信息查询 / 密码重置 / 权限 / 状态管理” 全生命周期，
# 是 routers/user.py 接口的直接支撑，核心聚焦 “用户数据的安全操作 + 数据库持久化”，逻辑严谨且兼顾安全性
# """


# 用户注册 / 登录 / 重置密码时，对明文密码做 SHA256 哈希加密，避免数据库存储明文密码（安全核心）
def password_hash(password: str) -> str:
    """对密码进行哈希处理。"""
    salt = "stock_agent_2026_salt"  # 给密码 “加的一把料”，本质是一段随机 / 固定的字符串，和明文密码拼接后再做哈希加密，目的是防止彩虹表破解
    return hashlib.sha256((password + salt).encode()).hexdigest()


# 为注册 / 登录 / 修改信息等操作提供 “用户存在性校验”，是所有用户操作的基础前提
def check_user_exists(username: str) -> bool:
    """检查用户是否存在。"""
    try:
        with SessionLocal() as session:
            user = session.query(UserTable).filter(UserTable.user_name == username).first()
            return user is not None

    except Exception as e:
        traceback.print_exc()
        return False


# 用户名唯一校验 → 密码加密 → 入库
def user_register(user_name: str, password: str, user_role: str) -> bool:
    try:
        with SessionLocal() as session:
            # 1. 校验：用户名是否已存在（防重复注册）
            user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user is not None:
                return False

            # 2. 密码加密
            password = password_hash(password)
            # 3. 创建用户记录（默认status=True，账号启用）
            user = UserTable(
                user_name=user_name,
                password=password,
                user_role=user_role,
                status=True,
                register_time=datetime.utcnow()
            )
            session.add(user)
            session.commit()
            return True

    except Exception as e:
        traceback.print_exc()
        return False


# 转换为 User 传输模型时，刻意隐藏了 password 字段，避免敏感信息泄露给前端
def get_user_info(user_name: str) -> Optional[User]:
    try:
        with SessionLocal() as session:
            user_db_record: UserTable | None = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user_db_record:
                # ORM模型 → 传输模型（隐藏password，仅返回前端需要的字段）
                return User(
                    user_id=user_db_record.id,
                    user_name=user_db_record.user_name,
                    user_role=user_db_record.user_role,
                    register_time=user_db_record.register_time,
                    status=user_db_record.status
                )
            else:
                return None
    except Exception as e:
        traceback.print_exc()
        return None


# 供管理员查看系统所有用户（分页避免数据量过大）
def list_users(page_index: int = 1, page_size: int = 200) -> List[User]:
    try:
        with SessionLocal() as session:
            page_index = max(page_index, 1)
            page_size = max(page_size, 1)
            # 分页查询：offset(跳过条数) + limit(每页条数)
            user_db_records = session.query(UserTable).offset((page_index - 1) * page_size).limit(page_size).all()
            # 批量转换为传输模型（同样隐藏password）
            return [User(user_id=user_db_record.id,
                    user_name=user_db_record.user_name,
                    user_role=user_db_record.user_role,
                    register_time=user_db_record.register_time,
                    status=user_db_record.status) for user_db_record in user_db_records]
    except Exception as e:
        traceback.print_exc()
        return []


# 用户存在性校验 → 密码加密对比 → 额外校验账号状态（启用/禁用）
def user_login(username: str, password: str) -> bool:
    try:
        with SessionLocal() as session:
            # 1. 校验：用户是否存在
            user = session.query(UserTable).filter(UserTable.user_name == username).first()
            if user is None:
                return False

            if not user.status:
                return False

            # 2. 密码校验：明文密码加密后与数据库存储的哈希值对比
            password = password_hash(password)
            if user.password != password:
                return False

            return True
    except Exception as e:
        traceback.print_exc()
        return False


def user_delete(user_name: str) -> bool:
    try:
        with SessionLocal() as session:
            user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user is None:
                return False

            session.delete(user)
            session.commit()
            return True
    except Exception as e:
        traceback.print_exc()
        return False


# 重置密码
def user_reset_password(user_name: str, password: str) -> bool:
    try:
        with SessionLocal() as session:
            user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user is None:
                return False
            # 新密码加密后更新
            user.password = password_hash(password)  # type: ignore
            session.commit()
            return True
    except Exception as e:
        traceback.print_exc()
        return False


# 管理员禁用 / 启用用户账号（比如禁用违规用户）
def alter_user_status(user_name: str, status: bool) -> bool:
    try:
        with SessionLocal() as session:
            user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user is None:
                return False
            user.status = status  # type: ignore
            session.commit()
            return True
    except Exception as e:
        traceback.print_exc()
        return False


# 修改用户角色  管理用户权限（比如普通用户升级为管理员）
def alter_user_role(user_name: str, user_role: str) -> bool:
    try:
        with SessionLocal() as session:
            user = session.query(UserTable).filter(UserTable.user_name == user_name).first()
            if user is None:
                return False

            valid_roles = ["admin", "user"]
            user.user_role = valid_roles  # type: ignore
            session.commit()
            return True
    except Exception as e:
        traceback.print_exc()
        return False

