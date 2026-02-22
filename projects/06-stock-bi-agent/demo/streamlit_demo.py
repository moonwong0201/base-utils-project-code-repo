# 导入 Streamlit 库
import streamlit as st

# """
# 整个 Streamlit 应用的启动文件，所有用户交互（登录、股票查询、智能对话等）都从这里开始。
# """

if "logged" not in st.session_state:
    st.session_state.logged = False

# --- 账户中心 ---
page_user_register = st.Page("user/user_register.py", title="用户注册", icon="➕")
page_user_login = st.Page("user/user_login.py", title="登陆/退出", icon="🚪")
page_user_info = st.Page("user/user_info.py", title="个人信息", icon="👤")
page_user_reset = st.Page("user/user_reset.py", title="修改信息", icon="✏️")
page_user_delete = st.Page("user/user_delete.py", title="删除账户", icon="❌")
page_user_list = st.Page("user/user_list.py", title="列举账户", icon="👥")

# --- 股票中心 ---
page_stock_search = st.Page("stock/stock_search.py", title="股票搜索", icon="🔍")
page_stock_industry = st.Page("stock/stock_industry.py", title="行业概览", icon="🏷️")
page_stock_board = st.Page("stock/stock_board.py", title="市场看板", icon="🧩")
page_stock_rank = st.Page("stock/stock_rank.py", title="股票排行", icon="🏆")
page_stock_info = st.Page("stock/stock_info.py", title="股票信息", icon="ℹ️")
page_stock_kline = st.Page("stock/stock_kline.py", title="股票K线图", icon="📊")
page_stock_min = st.Page("stock/stock_min_data.py", title="当日交易", icon="📊")
page_stock_fav = st.Page("stock/stock_favorite.py", title="股票收藏", icon="⭐")

# --- 数据中心 ---
page_data_list = st.Page("data/data_list.py", title="数据列表", icon="📑")
page_data_manage = st.Page("data/data_manage.py", title="数据管理", icon="⚙️")

# --- 智能问答 ---
page_chat_list = st.Page("chat/chat_list.py", title="对话历史", icon="🕰️")
page_chat = st.Page("chat/chat.py", title="通用对话", icon="💬")

mcp_list = st.Page("mcp/mcp_list.py", title="MCP列表", icon="⚙️")
mcp_debug = st.Page("mcp/mcp_debug.py", title="MCP调试", icon="🐞")


if st.session_state.logged:
    pg = st.navigation(  # 生成侧边栏菜单
        {
            "账户中心": [page_user_login, page_user_info, page_user_reset, page_user_delete, page_user_list],
            "股票中心": [page_stock_search, page_stock_board, page_stock_industry, page_stock_rank, page_stock_info, page_stock_kline, page_stock_min, page_stock_fav],
            # "数据中心": [page_data_list, page_data_manage],
            "工具中心": [mcp_list, mcp_debug],
            "智能问答": [page_chat_list, page_chat],
        }
    )
else:
    pg = st.navigation(
        {
            "账户中心": [page_user_register, page_user_login],
            "股票中心": [page_stock_search, page_stock_board, page_stock_industry, page_stock_rank, page_stock_info, page_stock_kline, page_stock_min, page_stock_fav],
            "智能问答": [page_chat_list, page_chat],
        }
    )

pg.run()
