import streamlit as st
import time, requests

# """
# 用户列表查询页面（管理员专属），核心功能是「已登录用户先查询自身信息，
# 若为管理员则调用后端接口查询全量用户列表并展示，普通用户则提示无权限」——
# 是用户管理模块的权限管控核心功能，实现 “管理员查看所有用户、普通用户无权限” 的分级管理
# """

def get_user(user_name):
    response = requests.post(
        "http://127.0.0.1:8000/v1/users/info",
        params={"user_name": user_name}
    ).json()

    if response["data"]["user_role"] == "管理员":
        response = requests.post(
            "http://127.0.0.1:8000/v1/users/list",
            params={"user_name": user_name}
        ).json()

        st.dataframe(response["data"])
    else:
        st.write("您是普通用户, 无权查看其他用户信息！")

    if response['code'] == 200:
        return True
    else:
        return False

if st.session_state.get('logged', False):
    get_user(st.session_state['user_name'])
