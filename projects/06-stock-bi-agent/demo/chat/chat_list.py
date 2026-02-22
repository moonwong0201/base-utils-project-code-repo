import streamlit as st
import requests

# """
# 基于 Streamlit 构建的 “聊天会话列表” 前端 Demo，核心功能是展示当前登录用户的所有历史聊天会话，
# 并提供 “进入聊天”“删除聊天” 的操作入口，是 Demo 中 “会话管理” 的核心页面，
# 衔接了后端 services/chat.py 的 list_chat/delete_chat_session 函数
# """

if st.session_state.get('logged', False):
    st.sidebar.markdown(f"用户名：{st.session_state['user_name']}")

    # 调用后端接口获取当前用户的所有会话
    data = requests.post("http://127.0.0.1:8000/v1/chat/list", params={"user_name": st.session_state['user_name']})
    response_data = data.json()
    if response_data.get("code") != 200:
        st.error(f"获取会话列表失败: {response_data.get('message')}")
        chat_data = []
    else:
        chat_data = response_data.get("data", [])[::-1]  # 倒序排列（最新的会话在最前面）

    # 为每个聊天会话创建卡片式展示
    for chat in chat_data:
        with st.container():   # 每个会话一个独立容器，保证样式隔离
            # 分三列布局：会话信息（3份）、反馈（2份）、操作按钮（1份）
            col1, col2, col3 = st.columns([3, 2, 1])

            # 第一列：会话核心信息（ID/标题/创建时间）
            with col1:
                st.markdown(f"**{chat["session_id"]} / {chat['title']}**")  # 会话ID + 标题（用户首个问题）
                st.caption(f"创建时间: {chat['start_time']}")  # 小号字体显示创建时间

            # 第二列：反馈信息（暂无/有反馈）
            with col2:
                feedback_text = "暂无反馈" if chat['feedback'] is None else chat['feedback']
                st.text(f"反馈: {feedback_text}")

            # 第三列：操作按钮（进入聊天/删除聊天）
            with col3:
                # 使用HTML a标签实现页面内跳转
                session_id = chat['session_id']
                # st.session_state.session_id = session_id  # 把当前会话ID存入session_state

                if st.button("进入聊天", key=session_id + "chat"):
                    st.session_state.session_id = session_id
                    st.session_state.loaded_session_id = None
                    st.session_state.messages = []
                    st.session_state.from_history = True

                    st.switch_page("chat/chat.py")

                # 删除聊天按钮：调用后端删除接口，删除后刷新页面
                if st.button("删除聊天", key=session_id + "del"):
                    requests.post("http://127.0.0.1:8000/v1/chat/delete", params={"session_id": session_id})
                    st.rerun()

            st.divider()   # 会话卡片之间加分隔线，提升可读性

else:
    st.info("请先登录再使用模型～")

# 前端逻辑	    调用的后端接口 / 模块	                核心交互内容
# 拉取会话列表	routers/chat.py → /v1/chat/list	    传入用户名，获取该用户的所有会话（session_id / 标题 / 创建时间）
# 进入聊天	    跳转到 chat/chat.py                  把选中的 session_id 存入 st.session_state，chat.py 会根据该 ID 拉取历史对话
# 删除聊天	    routers/chat.py → /v1/chat/delete	传入 session_id，调用 services/chat.py 的 delete_chat_session 删除会话及消息
