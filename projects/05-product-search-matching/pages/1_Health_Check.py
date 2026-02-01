import streamlit as st
import requests

st.title("🩺 1. 服务健康检查")

HEALTH_URL = "http://localhost:8000/health"
headers = {'accept': 'application/json'}

if st.button("运行健康检查"):  # 按钮触发接口调用
    try:
        response = requests.get(HEALTH_URL, headers=headers)

        st.subheader("API响应")
        st.code(f"URL: {HEALTH_URL}", language='http')

        st.metric(label="接口状态码", value=response.status_code)

        try:
            st.json(response.json())
        except requests.exceptions.JSONDecodeError:
            st.warning("后端返回非JSON格式响应，显示原始文本：")
            st.code(response.text, language='text')

        if response.status_code == 200:
            st.success("✅ 服务运行正常 (Status: 200 OK)")
        else:
            st.error(f"❌ 服务异常 (Status: {response.status_code})")

    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到 FastAPI 服务！\n请确认：\n1. 服务已启动\n2. 地址是 http://localhost:8000")
    except Exception as e:
        st.error("❌ 健康检查过程中发生未知错误：")
        st.exception(e)  # 展示完整的异常堆栈信息
