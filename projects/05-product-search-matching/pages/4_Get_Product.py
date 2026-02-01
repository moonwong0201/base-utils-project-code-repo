import streamlit as st
import requests

st.title("🔍 4. 获取单个商品详情")

# FastAPI 基础 URL
API_BASE_URL = "http://127.0.0.1:8000"
PRODUCT_URL = f"{API_BASE_URL}/product/"

# 假设您的 FastAPI 有一个端点用于提供图片文件
# 例如：如果 image_path 是 ./product_images/3.jpeg，那么图片访问路径是 /images/product_images/3.jpeg
# 您需要根据您后端实际的图片服务接口进行调整。
IMAGE_SERVE_ENDPOINT = "images" 

product_id = st.number_input("输入商品 ID", min_value=1, step=1, value=1)

if st.button("获取商品"):
    url = f"{PRODUCT_URL}{product_id}"
    try:
        response = requests.get(url)
        
        st.subheader("API 响应")
        st.code(f"URL: {url}", language='http')
        st.metric(label="状态码", value=response.status_code)
        
        if response.status_code == 200:
            st.success("✅ 成功获取商品详情。")
            data = response.json()
            st.json(data)
            st.image(data["data"]['image_path'], caption=data.get('title', '商品图片'), use_column_width=True)
            
        elif response.status_code == 404:
            st.warning("商品不存在 (Status: 404 Not Found)。")
            st.code(response.text, language='json')
        else:
            st.error(f"获取商品失败 (Status: {response.status_code})")
            st.code(response.text, language='json')

    except requests.exceptions.ConnectionError:
        st.error("无法连接到 FastAPI 服务。")
    except Exception as e:
        st.exception(e)