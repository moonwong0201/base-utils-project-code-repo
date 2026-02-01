# pages/Search_Function.py (已更新支持四种模式和Base64图片上传 + 结果图片展示)
import streamlit as st
import requests
import json
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import traceback

# --- 配置 (本地定义) ---
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="独立检索页面",
    layout="wide",
)

st.title("🔍 产品语义检索功能 (支持四种模式)")
st.markdown(f"**API 端点:** `{API_BASE_URL}/product/search`")
st.markdown("通过文本或图片查询，在产品库中执行语义相似度搜索。")

# --- 辅助函数：将上传文件转换为 Base64 ---
def get_image_base64(uploaded_file):
    """将上传的文件转换为 Base64 字符串。"""
    if uploaded_file is None:
        return None
    bytes_data = uploaded_file.getvalue()
    base64_data = base64.b64encode(bytes_data).decode('utf-8')
    return base64_data

# --- 检索操作函数 (本地定义) ---

def search_products_api(search_type, query_text, query_image_base64, top_k):
    """
    调用 API 的 /product/search 接口进行检索。
    返回 (原始结果列表, 成功标志)
    """
    url = f"{API_BASE_URL}/product/search"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # 动态构建请求体
    data = {
        "search_type": search_type,
        "top_k": top_k
    }

    if search_type in ["text2text", "text2image"]:
        if not query_text:
            st.error(f"在 {search_type} 模式下，查询文本不能为空。")
            return [], False
        data["query_text"] = query_text
    
    elif search_type in ["image2text", "image2image"]:
        if not query_image_base64:
            st.error(f"在 {search_type} 模式下，必须上传查询图片。")
            return [], False
        data["query_image"] = query_image_base64

    st.info(f"发送检索请求 (Type: **{search_type}**)...")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        # 假设所有成功的 API 调用都返回 JSON，并包含 status/message/data 字段
        response_json = response.json()
        
        # 检查后端返回的状态码
        if response_json.get("status") != 200:
            st.error(f"检索失败 (HTTP {response.status_code}): {response_json.get('message', '无详细信息')}")
            return [], False
        
        # 成功处理
        return response_json.get("data", []), True
            
    except requests.exceptions.ConnectionError:
        st.error(f"🚨 **连接错误：** 无法连接到 {API_BASE_URL}。请检查您的本地服务是否运行。")
    except Exception as e:
        traceback.print_exc()
        st.error(f"检索时发生未知错误: {e}")
        
    return [], False

# --- Streamlit 界面 ---

with st.form("search_form"):
    
    st.subheader("参数配置")
    
    # 检索类型选择
    search_type = st.radio(
        "选择检索类型 (`search_type`)",
        ("text2text", "text2image", "image2text", "image2image"),
        index=0,
    )
    
    # --- 动态输入区域 ---
    query_text = ""
    query_image = None
    query_image_base64 = None

    if search_type in ["text2text", "text2image"]:
        # 文本查询模式
        query_text = st.text_input(
            "查询文本 (`query_text`)", 
            placeholder="例如：机器学习课程 或 可爱的皮卡丘"
        )
    
    elif search_type in ["image2text", "image2image"]:
        # 图片查询模式
        query_image = st.file_uploader(
            "上传查询图片 (`query_image`)",
            type=['png', 'jpg', 'jpeg']
        )
        if query_image:
            query_image_base64 = get_image_base64(query_image)
            st.success("图片已上传并转换为 Base64。")
            
            # 显示查询图片预览
            st.image(query_image, caption="查询图片预览", width=150)


    # Top K 选择
    top_k = st.slider(
        "返回结果数量 (`top_k`)",
        min_value=1,
        max_value=20,
        value=10
    )
    
    submitted = st.form_submit_button("🚀 执行检索")

st.markdown("---")

if submitted:
    
    # 执行检索
    results_list, success = search_products_api(search_type, query_text, query_image_base64, top_k)
    
    st.subheader("📊 检索结果")
    
    if success:
        if results_list:
            st.success(f"✅ 检索成功！找到 {len(results_list)} 个相关产品。")
            
            # 使用 st.columns 迭代展示结果，实现图片和信息的并排显示
            st.markdown("### 🔍 详细结果展示")
            
            for i, item in enumerate(results_list):
                # 创建两列布局：左边放图片，右边放文本信息
                col1, col2 = st.columns([1, 4]) 
                
                # 左列：图片
                with col1:
                    image_path = item.get('image_path', None)
                    if image_path:
                        # 假设 image_path 是 Streamlit 可以直接访问的 URL 或路径
                        # 注意：如果 image_path 是本地文件系统路径，Streamlit 可能无法直接访问，
                        # 需要确保后端将图片暴露为静态资源，并返回完整的 URL。
                        st.image(image_path, caption=f"ID: {item.get('id', 'N/A')}", use_container_width='always')
                    else:
                        st.warning("无图片路径")

                # 右列：文本信息
                with col2:
                    st.markdown(f"**排名:** #{i + 1}")
                    st.markdown(f"**标题:** `{item.get('title', 'N/A')}`")
                    st.markdown(f"**相似度 (Distance):** `{item.get('distance', 0):.4f}`")
                    st.markdown(f"**Milvus Key:** `{item.get('milvus_primary_key', 'N/A')}`")
                    st.markdown(f"**创建/更新时间:** {item.get('created_at', 'N/A')} / {item.get('updated_at', 'N/A')}")
                    
                st.markdown("---") # 分隔线
        else:
            st.info("ℹ️ 未找到符合条件的产品。")