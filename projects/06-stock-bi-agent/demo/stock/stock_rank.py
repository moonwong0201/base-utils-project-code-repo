import streamlit as st
import requests
import pandas as pd
from typing import Optional

# """
# 股票排行榜页面，核心功能是「通过多维度筛选（市场板块 / 行业 / 排序字段 / 排序方式）+ 分页，
# 调用后端接口获取股票排行数据并可视化展示」—— 是股票模块中 “批量股票筛选分析” 的核心页面，
# 支持从全市场 / 指定行业中按价格、涨跌幅、成交量等维度排序，满足 “找龙头、找异动股” 的批量分析需求
# """

# -------------------- API 配置 --------------------
BASE_URL = "http://127.0.0.1:8000"
RANK_ENDPOINT = "/stock/get_stock_rank"
# --------------------------------------------------

if st.session_state.get('logged', False):
    st.sidebar.markdown(f"用户名：{st.session_state['user_name']}")

# 定义选项映射
NODE_OPTIONS = {
    "沪深A股 (a)": 'a',
    "上交所A股 (ash)": 'ash',
    "深交所A股 (asz)": 'asz',
    "上交所B股 (bsh)": 'bsh',
    "深交所B股 (bsz)": 'bsz',
}

SORT_OPTIONS = {
    "交易价格": 'price',
    "涨跌额": 'priceChange',
    "涨跌幅": 'changePercent',
    "成交量": 'volume',
    "成交额": 'turnover',
    "今开盘": 'open',
    "最高价": 'high',
    "最低价": 'low',
    # 更多字段可以根据需要添加...
}


# -------------------- API 调用函数 --------------------

def fetch_stock_rank(
        node: str,
        industry_code: Optional[str] = None,
        page_index: int = 1,
        page_size: int = 100,
        sort_field: str = "price",
        asc: int = 0
):
    """
    通过调用后端 API 获取股票排行数据。
    """
    url = f"{BASE_URL}{RANK_ENDPOINT}"

    params = {
        "node": node,
        "pageIndex": page_index,
        "pageSize": page_size,
        "sort": sort_field,
        "asc": asc,
    }
    if industry_code:
        params["industryCode"] = industry_code

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("code") == 200 and data.get("data"):
            return data["data"]
        else:
            st.warning(f"API 返回成功，但数据为空或不符合预期。")
            return None

    except requests.exceptions.ConnectionError:
        st.error(f"连接错误：无法连接到后端服务 ({BASE_URL})。请确保后端服务正在运行。")
        return None
    except Exception as e:
        st.error(f"发生未知错误：{e}")
        return None


# -------------------- Streamlit 页面 --------------------

def stock_rank_page():

    # -------------------- 1. 参数选择侧边栏 --------------------
    st.sidebar.header("排行榜筛选条件")

    # 1.1 市场/板块选择 (Node)
    selected_node_name = st.selectbox(
        "选择市场/板块",
        options=list(NODE_OPTIONS.keys()),
        index=0,
        key="rank_node_select"
    )
    selected_node = NODE_OPTIONS[selected_node_name]

    # 1.2 排序字段选择 (Sort)
    selected_sort_name = st.selectbox(
        "选择排序字段",
        options=list(SORT_OPTIONS.keys()),
        index=0,
        key="rank_sort_select"
    )
    selected_sort = SORT_OPTIONS[selected_sort_name]

    # 1.3 排序方式 (Asc)
    sort_asc = st.radio(
        "排序方式",
        options=["降序 (高到低)", "升序 (低到高)"],
        index=0,
        key="rank_asc_select"
    )
    selected_asc = 1 if sort_asc == "升序 (低到高)" else 0

    # 1.4 行业代码筛选 (Industry Code)
    # 理想情况下，这里应该从 `stock_industry.py` 获取完整的行业列表
    industry_code = st.text_input(
        "行业代码筛选 (可选)",
        placeholder="例如: sw_dz (电子)",
        key="rank_industry_code",
        value="sw_sysh"
    ).strip()

    # 1.5 分页大小
    page_size = st.slider("每页显示数量", min_value=10, max_value=200, value=50, step=10)

    st.sidebar.markdown("---")

    # -------------------- 2. 主页面数据展示 --------------------

    # 初始化当前页码
    if 'rank_page_index' not in st.session_state:
        st.session_state['rank_page_index'] = 1

    # 初始化：默认不查询
    if 'should_fetch' not in st.session_state:
        st.session_state['should_fetch'] = False

    # 按钮：触发数据加载
    if st.button("🔎 查询排行榜"):
        st.session_state['should_fetch'] = True
        st.session_state['rank_page_index'] = 1
        st.rerun()  # 立即重新运行，显示结果

    # 只有点击过按钮才查询
    if not st.session_state['should_fetch']:
        st.info("请选择条件并点击「查询排行榜」")
        return  # 直接返回，不执行后续查询

    # 重新加载或第一次加载
    current_page = st.session_state['rank_page_index']

    # 调用 API 获取数据
    rank_data = fetch_stock_rank(
        node=selected_node,
        industry_code=industry_code if industry_code else None,
        page_index=current_page,
        page_size=page_size,
        sort_field=selected_sort,
        asc=selected_asc
    )

    if rank_data is None:
        st.stop()  # 停止执行后续代码

    # 提取核心数据
    total_records = rank_data.get('totalRecord', 0)
    rank_list = rank_data.get('rank', [])

    # -------------------- 3. 统计和分页控件 --------------------

    max_pages = (total_records + page_size - 1) // page_size
    start_record = (current_page - 1) * page_size + 1
    end_record = min(current_page * page_size, total_records)

    st.info(
        f"📈 找到 **{total_records}** 条记录。当前显示第 **{start_record}** 到 **{end_record}** 条 (第 {current_page} / {max_pages} 页)。")

    # 分页控制按钮
    col_prev, col_page_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        if current_page > 1 and st.button("上一页"):
            st.session_state['rank_page_index'] -= 1
            st.rerun()

    with col_page_info:
        st.markdown(f"<p style='text-align: center; font-size: 16px;'>当前页: {current_page}</p>",
                    unsafe_allow_html=True)

    with col_next:
        if current_page < max_pages and st.button("下一页"):
            st.session_state['rank_page_index'] += 1
            st.rerun()

    st.markdown("---")

    # -------------------- 4. 排行榜数据表格 --------------------

    if rank_list:
        df_rank = pd.DataFrame(rank_list)

        # 简化和重命名列
        df_display = df_rank[[
            'code', 'name', 'price', 'priceChange', 'changePercent',
            'volume', 'turnover', 'open', 'high', 'low', 'date'
        ]].copy()

        df_display.columns = [
            '代码', '名称', '最新价', '涨跌额', '涨跌幅(%)',
            '成交量', '成交额(万)', '今开', '最高', '最低', '更新时间'
        ]

        # 应用颜色样式
        def color_rank_changes(val):
            """根据涨跌幅应用颜色"""
            if pd.isna(val):
                return ''
            try:
                # 假设涨跌幅和涨跌额都是数字或数字字符串
                val_float = float(str(val).strip('%'))
                if val_float > 0:
                    color = 'red'
                elif val_float < 0:
                    color = 'green'
                else:
                    color = 'black'
                return f'color: {color}; font-weight: bold'
            except ValueError:
                return ''

        st.dataframe(
            df_display.style.applymap(color_rank_changes, subset=['涨跌额', '涨跌幅(%)']),
            hide_index=True,
            use_container_width=True
        )


if __name__ == '__main__':
    stock_rank_page()