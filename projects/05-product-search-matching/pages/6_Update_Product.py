import streamlit as st
import requests

st.title("✏️ 6. 更新商品信息 (PATCH)")

base_url = "http://127.0.0.1:8000/product/"

product_id_to_update = st.number_input("输入要更新的商品 ID", min_value=1, step=1, key='update_id')

st.header("1. 更新商品标题")

with st.form("update_title_form"):
    new_title = st.text_input("新标题", key='new_title')
    title_submitted = st.form_submit_button("更新标题")
    
    if title_submitted and product_id_to_update and new_title:
        url = f"{base_url}{product_id_to_update}/title"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {'title': new_title}
        
        try:
            response = requests.patch(url, headers=headers, data=data)
            st.subheader("API 响应")
            st.code(f"URL: {url}", language='http')
            st.metric(label="状态码", value=response.status_code)
            
            if response.status_code == 200:
                st.success(f"✅ 商品 ID **{product_id_to_update}** 标题更新成功!")
                st.json(response.json())
            else:
                st.error(f"标题更新失败 (Status: {response.status_code})")
                st.code(response.text, language='json')
        except Exception as e:
            st.exception(e)

st.header("2. 更新商品图片")

with st.form("update_image_form"):
    uploaded_file = st.file_uploader("上传新图片 (JPEG/PNG)", type=['jpg', 'jpeg', 'png'], key='new_image')
    image_submitted = st.form_submit_button("更新图片")
    
    if image_submitted and product_id_to_update and uploaded_file:
        url = f"{base_url}{product_id_to_update}/image"
        
        files = {
            'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        
        try:
            response = requests.patch(url, files=files)
            st.subheader("API 响应")
            st.code(f"URL: {url}", language='http')
            st.metric(label="状态码", value=response.status_code)
            
            if response.status_code == 200:
                st.success(f"✅ 商品 ID **{product_id_to_update}** 图片更新成功!")
                st.json(response.json())
            else:
                st.error(f"图片更新失败 (Status: {response.status_code})")
                st.code(response.text, language='json')

        except Exception as e:
            st.exception(e)