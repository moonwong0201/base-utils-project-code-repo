import traceback

from fastapi import FastAPI, APIRouter  # type: ignore
from typing import List
import services.user as user_service
from models.data_models import (
    RequestForUserLogin,
    BasicResponse,
    RequestForUserRegister,
    RequestForUserResetPassword,
    RequestForUserChangeInfo,
    User
)

# 从 models.data_models 模块中导入多个 Pydantic 模型，用于请求参数的校验和响应格式的统一。
# RequestForUserLogin: 用于校验用户登录请求的参数（如用户名、密码）。
# RequestForUserRegister: 用于校验用户注册请求的参数（如用户名、密码、角色）。
# RequestForUserResetPassword: 用于校验密码重置请求的参数。
# RequestForUserChangeInfo: 用于校验用户信息修改请求的参数。
# BasicResponse: 所有接口的统一响应模型，包含 code (状态码)、message (提示信息) 和 data (业务数据)。

router = APIRouter(prefix="/v1/users", tags=["users"])
# tags=["users"]: 给这个路由模块打上 “users” 标签，在 FastAPI 自动生成的 API 文档中，所有用户相关的接口会被归类到一起

"""
定义一个 FastAPI 路由模块，专门处理与 “用户” 相关的 Web API 接口。
它提供了用户注册、登录、密码重置、信息查询与修改、用户删除以及用户列表查询等完整的用户管理功能
"""


# 用户登录 验证用户身份
@router.post("/login", response_model=BasicResponse[dict])
def user_login(req: RequestForUserLogin) -> BasicResponse:
    try:
        login_success = user_service.user_login(req.user_name, req.password)
        # 调用服务层函数验证用户名和密码
        if login_success:
            # 2. 登录成功后，调用get_user_info获取用户角色（关键适配步骤）
            user_detail = user_service.get_user_info(user_name=req.user_name)
            # 3. 组装返回数据（从user_detail中取user_role）
            return BasicResponse(
                code=200,
                message="用户登录成功",
                data={
                    "user_name": req.user_name,
                    "user_role": user_detail.user_role  # 从get_user_info的返回值中取
                }
            )
        else:
            return BasicResponse(code=401, message="用户名或密码错误/用户已禁用", data={})
    except Exception as e:
        error_msg = f"登录失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data={})


# 创建一个新用户账户
@router.post("/register", response_model=BasicResponse[bool])
def user_register(req: RequestForUserRegister) -> BasicResponse:
    try:
        register_result = user_service.user_register(req.user_name, req.password, req.user_role)
        # 调用服务层函数创建新用户
        if register_result:
            return BasicResponse(code=201, message="用户注册成功", data=register_result)
        else:
            return BasicResponse(code=409, message="用户名已存在", data=False)
    except Exception as e:
        error_msg = f"注册失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=False)


# 允许用户在验证旧密码后设置新密码
@router.post("/reset-password", response_model=BasicResponse[bool])
def user_reset_password(req: RequestForUserResetPassword) -> BasicResponse:
    try:
        # 先验证旧密码是否正确
        if not user_service.user_login(req.user_name, req.password):
            return BasicResponse(code=401, message="用户名或密码错误", data=False)
        else:
            reset_result = user_service.user_reset_password(req.user_name, req.new_password)
            # 调用服务层函数重置密码
            if reset_result:
                return BasicResponse(code=200, message="密码重置成功", data=reset_result)
            else:
                return BasicResponse(code=200, message="密码重置失败", data=False)
    except Exception as e:
        error_msg = f"重置密码失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=False)


# 根据用户名查询用户的详细信息
@router.post("/info", response_model=BasicResponse[User])
def user_info(user_name: str) -> BasicResponse:
    try:
        # 检查用户是否存在
        if not user_service.check_user_exists(user_name):
            return BasicResponse(code=400, message="用户不存在", data=None)
        else:
            user_detail = user_service.get_user_info(user_name=user_name)
            user_data = User(
                user_id=user_detail.user_id,
                user_name=user_detail.user_name,
                user_role=user_detail.user_role,
                register_time=user_detail.register_time,
                status=user_detail.status
            )
            # data: 调用服务层函数获取用户详细信息
            return BasicResponse(code=200, message="获取用户信息成功", data=user_data)
    except Exception as e:
        error_msg = f"获取用户信息失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=None)


# 修改用户的部分信息
@router.post("/reset-info", response_model=BasicResponse[bool])
def user_reset_info(req: RequestForUserChangeInfo) -> BasicResponse:
    try:
        if not user_service.check_user_exists(req.user_name):
            return BasicResponse(code=404, message="用户不存在", data=False)

        # 标记是否有实际修改操作
        has_update = False
        if req.user_role is not None:
            # 根据请求中的字段，选择性地修改用户信息
            user_service.alter_user_role(req.user_name, req.user_role)
            has_update = True

        if req.status is not None:
            user_service.alter_user_status(req.user_name, req.status)
            has_update = True

        if has_update:
            return BasicResponse(code=200, message="用户信息修改成功", data=True)
        else:
            return BasicResponse(code=400, message="未传入任何需要修改的信息", data=False)

    except Exception as e:
        error_msg = f"修改用户信息失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=False)


# 根据用户名删除一个用户账户
@router.post("/delete", response_model=BasicResponse[bool])
def user_delete(user_name: str) -> BasicResponse:
    try:
        if not user_service.check_user_exists(user_name):
            return BasicResponse(code=400, message="用户不存在", data=False)
        else:
            # 调用服务层函数删除用户
            delete_result = user_service.user_delete(user_name)
            if delete_result:
                return BasicResponse(code=200, message="用户删除成功", data=True)
            else:
                return BasicResponse(code=500, message="用户删除失败", data=False)

    except Exception as e:
        error_msg = f"删除用户失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=False)


# 查询系统中所有用户的列表（通常用于管理员功能）
@router.post("/list", response_model=BasicResponse[List[User]])
def user_list() -> BasicResponse:
    try:
        # 调用服务层函数获取所有用户的列表
        user_list_data = user_service.list_users()
        # 转换为User模型列表，过滤敏感字段
        result_data = [
            User(
                user_id=u.user_id,
                user_name=u.user_name,
                user_role=u.user_role,
                register_time=u.register_time,
                status=u.status
            ) for u in user_list_data
        ]
        return BasicResponse(code=200, message="查询所有用户成功", data=result_data)

    except Exception as e:
        error_msg = f"查询用户列表失败：{str(e)}"
        print(traceback.format_exc())
        return BasicResponse(code=500, message=error_msg, data=[])
