import streamlit as st

# 设置网页的「全局基础样式」，这行代码必须放在所有其他 Streamlit 代码之前
st.set_page_config(
    page_title="🚀 FastAPI 商品管理 Demo",
    page_icon="🛍️"
)

# 设置页面主标题
st.title("🛍️ FastAPI 商品管理系统 Demo")
# 在网页左侧侧边栏显示一行「绿色背景的成功提示文字」
st.sidebar.success("请在左侧导航栏选择操作。")
# st.sidebar：所有以 st.sidebar.xxx 开头的代码，都会渲染到左侧侧边栏（默认隐藏，点击左上角箭头展开）；
# success()：Streamlit 提供的「状态提示函数」，不同状态对应不同颜色：
# st.success()：绿色（成功 / 提示）；
# st.info()：蓝色（信息）；
# st.warning()：黄色（警告）；
# st.error()：红色（错误）。

# 展示 Markdown 格式的功能说明
st.markdown(
    """
    这是一个用于演示如何通过 **Streamlit** 界面来调用 **FastAPI** 商品管理 API 的应用。

    **功能列表:**
    - **服务健康检查**: 确保后端服务正常运行。
    - **商品列表**: 获取所有已创建的商品。
    - **创建商品**: 上传图片和标题来创建新商品。
    - **获取商品**: 根据 ID 查看单个商品详情。
    - **删除商品**: 根据 ID 删除商品。
    - **更新商品**: 修改商品的标题或图片。
    """
)
