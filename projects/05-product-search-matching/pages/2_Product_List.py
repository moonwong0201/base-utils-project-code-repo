import streamlit as st
import requests

st.title("📋 2. 商品列表")

PRODUCT_LIST_URL = "http://localhost:8000/product/list"
headers = {'accept': 'application/json'}

# 初始化 session_state（记住当前页码，避免每次重置为1）
if "page_index" not in st.session_state:
    st.session_state.page_index = 1

# 每页条数选择
page_size = st.selectbox("每页条数", options=[10, 20, 50], index=0, key="page_size")

# 分页按钮区域（上一页、下一页、首页、末页）
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 首页", use_container_width=True):
        st.session_state.page_index = 1
with col2:
    if st.button("⬅️ 上一页", use_container_width=True):
        if st.session_state.page_index > 1:
            st.session_state.page_index -= 1
with col3:
    if st.button("➡️ 下一页", use_container_width=True):
        st.session_state.page_index += 1  # 先+1，后面会判断是否超出总页数
with col4:
    if st.button("🔚 末页", use_container_width=True):
        # 先请求一次，拿到总页数，再跳末页（避免提前不知道总页数）
        pass

# 手动输入页码（兼容按钮）
page_index = st.number_input(
    "页码",
    min_value=1,
    value=st.session_state.page_index,
    step=1,
    key="manual_page"
)
# 同步手动输入到 session_state
st.session_state.page_index = page_index

# 按钮：获取商品列表
if st.button("获取商品列表", type="primary", use_container_width=True):
    try:
        params = {
            "page_index": st.session_state.page_index,
            "page_size": page_size
        }
        response = requests.get(PRODUCT_LIST_URL, headers=headers, params=params)

        st.subheader("API 响应")
        st.code(f"URL: {response.url}", language='http')
        st.metric(label="状态码", value=response.status_code)

        if response.status_code == 200:
            data = response.json()
            if not data or not data.get("data"):
                st.warning("未获取到有效商品数据")
                st.stop()

            products = data["data"]["products"]
            pagination = data["data"].get("pagination", {})
            total = pagination.get("total", 0)
            total_pages = pagination.get("total_pages", 0)

            # 修正页码：如果当前页 > 总页数，自动跳到最后一页
            if st.session_state.page_index > total_pages and total_pages > 0:
                st.session_state.page_index = total_pages
                st.warning(f"页码超出范围，已自动跳至最后一页（第{total_pages}页）")
                # 重新请求修正后的页码
                params["page_index"] = st.session_state.page_index
                response = requests.get(PRODUCT_LIST_URL, headers=headers, params=params)
                data = response.json()
                products = data["data"]["products"]
                pagination = data["data"].get("pagination", {})

            # 展示分页信息
            st.subheader("分页信息")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("当前页", st.session_state.page_index)
            with c2:
                st.metric("每页", page_size)
            with c3:
                st.metric("总条数", total)
            with c4:
                st.metric("总页数", total_pages)
            st.divider()

            # 展示商品列表
            st.subheader("商品列表")
            if products:
                st.dataframe(products, hide_index=True, use_container_width=True)
            else:
                st.info("当前页暂无商品，请切换页码或每页条数")
        else:
            st.error(f"获取失败 (Status: {response.status_code})")
            st.code(response.text, language='json')

    except requests.exceptions.ConnectionError:
        st.error("无法连接后端服务，请先启动 FastAPI")
    except Exception as e:
        st.exception(e)

# import streamlit as st
# import requests
#
# st.title("📋 2. 商品列表")
#
# PRODUCT_LIST_URL = "http://localhost:8000/product/list"
# headers = {'accept': 'application/json'}
#
# if st.button("获取所有商品列表"):
#     try:
#         response = requests.get(PRODUCT_LIST_URL, headers=headers)
#
#         st.subheader("API 响应")
#         st.code(f"URL: {PRODUCT_LIST_URL}", language='http')
#         st.metric(label="状态码", value=response.status_code)
#
#         if response.status_code == 200:
#             data = response.json()
#             # 显示为表格
#             st.dataframe(data["data"]["products"])
#         else:
#             st.error(f"获取列表失败 (Status: {response.status_code})")
#             st.code(response.text, language='json')
#
#     except requests.exceptions.ConnectionError:
#         st.error("无法连接到 FastAPI 服务。")
#     except Exception as e:
#         st.exception(e)