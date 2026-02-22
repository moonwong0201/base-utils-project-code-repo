import streamlit as st
import requests
import pandas as pd

# """
# 股票行业概览页面，核心功能是「可视化展示所有申万行业的代码和名称列表」——
# 调用后端 /stock/get_industry_code 接口获取行业基础数据，以表格形式展示，
# 是股票模块的 “行业字典入口”，为后续按行业筛选股票、查看行业走势提供基础支撑
# """

# -------------------- API 配置 --------------------
BASE_URL = "http://127.0.0.1:8000"
INDUSTRY_ENDPOINT = "/stock/get_industry_code"


if st.session_state.get('logged', False):
    st.sidebar.markdown(f"用户名：{st.session_state['user_name']}")
# --------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="正在从后端加载行业数据...")
def fetch_industry_codes():
    """
    通过调用后端 API 获取所有申万行业代码和名称。
    使用 st.cache_data 缓存结果，因为行业列表通常不会频繁变动。
    """
    url = f"{BASE_URL}{INDUSTRY_ENDPOINT}"

    try:
        # 发送 GET 请求
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get("code") == 200 and data.get("data"):
            return data["data"]
        else:
            st.warning(f"API 返回成功，但数据为空或不符合预期。")
            return []

    except requests.exceptions.ConnectionError:
        st.error(f"连接错误：无法连接到后端服务 ({BASE_URL})。请确保后端服务正在运行。")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求失败：{e}")
        return None
    except Exception as e:
        st.error(f"发生未知错误：{e}")
        return None


def stock_rank_page():
    """定义股票排行（行业概览）页面的布局和逻辑"""

    # 获取行业数据
    industry_list = fetch_industry_codes()

    if industry_list is None:
        # 错误信息已在 fetch_industry_codes 中显示
        return

    if not industry_list:
        st.warning("未能获取到行业数据。")
        return

    # 将结果转换为 DataFrame 以便处理
    df_industries = pd.DataFrame(industry_list)
    df_industries.rename(columns={'name': '行业名称', 'industryCode': '行业代码'}, inplace=True)

    st.dataframe(df_industries, hide_index=True, use_container_width=True)


if __name__ == '__main__':
    stock_rank_page()