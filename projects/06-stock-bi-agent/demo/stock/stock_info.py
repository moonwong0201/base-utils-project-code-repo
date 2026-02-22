import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# """
# 个股详情页面，核心功能是「输入股票代码查询并可视化展示个股的全维度信息」——
# 调用后端 /stock/get_stock_info 接口获取单只股票的实时行情、交易指标、财务估值、五档盘口等数据，
# 按 “核心行情→交易细节→财务估值→盘口” 分层展示，是股票模块的 “个股深度分析入口”
# """

# -------------------- API 配置 --------------------
BASE_URL = "http://127.0.0.1:8000"
INFO_ENDPOINT = "/stock/get_stock_info"

if st.session_state.get('logged', False):
    st.sidebar.markdown(f"用户名：{st.session_state['user_name']}")
# --------------------------------------------------


def fetch_stock_info(code: str):
    """
    通过调用后端 API 获取特定股票的详细信息。
    """
    if not code:
        return None

    # 构造完整的 API URL
    url = f"{BASE_URL}{INFO_ENDPOINT}"
    params = {"code": code.strip()}

    try:
        # 发送 GET 请求
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # 对 4xx 或 5xx 状态码抛出异常

        data = response.json()

        # 检查业务状态码和数据是否存在
        if data.get("code") == 200 and data.get("data") and len(data["data"]) > 0:
            return data["data"][0]  # 返回列表中的第一个对象
        else:
            st.warning(f"API 返回成功，但未找到代码 {code} 的数据或数据为空。")
            return None

    except requests.exceptions.ConnectionError:
        st.error(f"连接错误：无法连接到后端服务 ({BASE_URL})。请确保后端服务正在运行。")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
        return None
    except Exception as e:
        st.error(f"发生未知错误：{e}")
        return None


def display_stock_data(info: dict):
    """
    结构化地展示股票详细信息。
    """

    # -------------------- 1. 核心行情概览 --------------------
    st.subheader(f"📈 {info.get('name', 'N/A')} ({info.get('code', 'N/A')})")

    # 格式化日期时间
    try:
        dt_obj = datetime.strptime(info.get('date', ''), '%Y-%m-%d %H:%M:%S')
        last_update = dt_obj.strftime('%Y年%m月%d日 %H:%M:%S')
    except:
        last_update = "N/A"

    st.caption(f"最后更新时间：{last_update}")

    # 计算涨跌颜色
    change_percent = info.get('changePercent', '0.00')
    try:
        change_float = float(change_percent)
        if change_float > 0:
            color = 'red'
            arrow = '▲'
        elif change_float < 0:
            color = 'green'
            arrow = '▼'
        else:
            color = 'gray'
            arrow = '—'
    except ValueError:
        color = 'gray'
        arrow = '—'

    # 主价格和涨跌幅展示
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(f"**<p style='font-size: 24px; color: {color};'>{info.get('price', 'N/A')}</p>**", unsafe_allow_html=True)

    with col2:
        st.markdown(f"**<p style='font-size: 24px; color: {color};'>{arrow} {info.get('priceChange', 'N/A')} {change_percent}%</p>**", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- 2. 关键交易指标 --------------------
    st.subheader("交易细节")

    # 使用 DataFrame 或 metric 进行指标展示
    metrics_data = {
        "今开": info.get('open', 'N/A'),
        "昨收": info.get('close', 'N/A'),
        "最高": info.get('high', 'N/A'),
        "最低": info.get('low', 'N/A'),
        "成交量 (手)": info.get('volume', 'N/A'),
        "成交额 (万)": info.get('turnover', 'N/A'),
        "换手率 (%)": info.get('turnoverRate', 'N/A'),
        "量比": info.get('volumeRate', 'N/A'),
    }

    # 将字典转换为 DataFrame，便于 Streamlit 展示
    df_metrics = pd.DataFrame(list(metrics_data.items()), columns=['指标', '值'])

    # 在页面上以两列展示
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.dataframe(df_metrics.iloc[:4], hide_index=True, use_container_width=True)
    col_metric2.dataframe(df_metrics.iloc[4:], hide_index=True, use_container_width=True)

    st.markdown("---")

    # -------------------- 3. 财务与估值指标 --------------------
    st.subheader("财务与估值")
    col_pe, col_spe, col_pb, col_worth = st.columns(4)

    col_pe.metric("市盈率(PE)", info.get('pe', 'N/A'))
    col_spe.metric("静态市盈率(SPE)", info.get('spe', 'N/A'))
    col_pb.metric("市净率(PB)", info.get('pb', 'N/A'))
    col_worth.metric("总市值(亿)", info.get('totalWorth', 'N/A'))

    st.markdown("---")

    # -------------------- 4. 五档盘口 --------------------
    st.subheader("买卖五档盘口")

    # 提取买入和卖出数据
    buy_list = info.get('buy', [])
    sell_list = info.get('sell', [])

    # 构造盘口数据帧
    b_data = []
    for i in range(0, len(buy_list), 2):
        b_data.append([f"买{i // 2 + 1}", buy_list[i], buy_list[i + 1]])

    s_data = []
    for i in range(0, len(sell_list), 2):
        s_data.append([f"卖{len(sell_list) // 2 - i // 2}", sell_list[i], sell_list[i + 1]])  # 卖盘从卖1到卖5是倒序的
    s_data.reverse()  # 调整顺序使其显示为卖1, 卖2...

    df_buy = pd.DataFrame(b_data, columns=['档位', '价格', '数量(手)'])
    df_sell = pd.DataFrame(s_data, columns=['档位', '价格', '数量(手)'])

    col_sell, col_buy = st.columns(2)

    with col_sell:
        st.markdown("**卖盘 (Sell)**")
        st.dataframe(df_sell, hide_index=True, use_container_width=True)

    with col_buy:
        st.markdown("**买盘 (Buy)**")
        st.dataframe(df_buy, hide_index=True, use_container_width=True)

    st.json(info)

def stock_info_page():
    """页面主入口函数"""

    # 使用 Session State 记住上次查询的代码
    if 'last_stock_code' not in st.session_state:
        st.session_state['last_stock_code'] = "sz002392"  # 默认值

    # 输入区域
    with st.form(key='stock_info_form'):
        stock_code = st.text_input(
            "请输入股票代码",
            value=st.session_state['last_stock_code'],
            placeholder="例如：sh600519 或 sz002392"
        )
        submitted = st.form_submit_button("查询实时信息")

    if submitted or st.session_state['last_stock_code'] != stock_code:
        if stock_code:
            st.session_state['last_stock_code'] = stock_code

            # 使用 Spinner 显示加载状态
            with st.spinner(f"正在查询股票 {stock_code} 的实时数据..."):
                stock_info = fetch_stock_info(stock_code)

                if stock_info:
                    display_stock_data(stock_info)
        else:
            st.warning("请输入有效的股票代码进行查询。")


if __name__ == '__main__':
    stock_info_page()
