import streamlit as st
import time, requests

# """
# 用户信息查询页面，核心功能是「已登录用户自动调用后端接口查询当前账号信息，并展示原始响应结果」
# —— 是用户管理模块的基础功能，用于查看账号详情，为后续的 “修改信息 / 重置密码” 等功能提供数据支撑
# """

def get_user(user_name):
    response = requests.post(
        "http://127.0.0.1:8000/v1/users/info",
        params={"user_name": user_name}
    ).json()

    st.write(response)

    if response['code'] == 200:
        return True
    else:
        return False

if st.session_state.get('logged', False):
    get_user(st.session_state['user_name'])
