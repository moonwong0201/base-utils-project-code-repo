# -*- coding: utf-8 -*-
"""
公式智能问答系统 - Streamlit前端
运行命令：streamlit run streamlit_ui.py
"""
import streamlit as st
import requests
import json

# 后端API
BACKEND_API_URL = "http://localhost:8000/qa"


# 调用后端接口函数
def call_qa_api(user_question: str):
    """调用后端API，返回结果"""
    try:
        # 发送POST请求到后端URL
        response = requests.post(
            BACKEND_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": user_question})
        )
        # 解析响应
        if response.status_code == 200:
            return response.json()
        else:
            return {"code": response.status_code, "error": f"接口调用失败：{response.text}"}
    except Exception as e:
        return {"code": 500, "error": f"请求后端出错：{str(e)}（请检查后端是否启动）"}


# 前端UI
def main():
    # 页面基础配置
    st.set_page_config(
        page_title="公式智能问答系统",
        page_icon="🧮",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # 页面标题与说明
    st.title("🧮 公式智能问答系统")
    st.markdown("#### 支持农业、工程、电商等多行业公式的高精度计算与自然语言问答")
    st.divider()

    # 输入区域
    user_question = st.text_area(
        label="请输入你的问题",
        placeholder="例如：计算农产品在零售价格100，生产成本80，日销量100下的利润",
        height=100,
        key="user_input"
    )

    # 提交按钮
    submit_btn = st.button("🚀 提交计算", type="primary")

    # 输出区域
    if submit_btn:
        if not user_question.strip():
            st.warning("❌ 请输入有效的计算问题！")
        else:
            # 加载中状态
            with st.spinner("正在匹配工具并计算..."):
                # 调用后端API
                result = call_qa_api(user_question)

                # 展示结果
                if result["code"] == 200:
                    st.subheader("📝 匹配的核心工具")
                    st.write(", ".join(result["matched_tools"]))
                    st.divider()
                    st.subheader("💡 计算结果")
                    st.write(result["answer"])
                else:
                    st.error(f"❌ 处理失败：{result['error']}")


if __name__ == "__main__":
    main()
