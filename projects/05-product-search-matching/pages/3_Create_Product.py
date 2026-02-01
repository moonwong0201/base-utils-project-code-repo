import streamlit as st
import requests

st.title("➕ 3. 创建新商品")

CREATE_URL = "http://127.0.0.1:8000/product"

with st.form("create_product_form"):
    title = st.text_input("商品标题", value='神奇宝贝')
    uploaded_file = st.file_uploader("上传商品图片 (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    submitted = st.form_submit_button("创建商品")
    
    if submitted:
        if not title:
            st.warning("请输入商品标题。")
        elif not uploaded_file:
            st.warning("请上传商品图片。")
        else:
            # 准备请求数据
            files = {
                'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            data = {'title': title}
            
            try:
                with st.spinner('正在创建商品...'):
                    response = requests.post(CREATE_URL, files=files, data=data)
                
                st.subheader("API 响应")
                st.code(f"URL: {CREATE_URL}", language='http')
                st.metric(label="状态码", value=response.status_code)
                
                if response.status_code == 200:
                    st.success("🎉 商品创建成功!")
                    st.json(response.json())
                else:
                    st.error(f"创建商品失败 (Status: {response.status_code})")
                    st.code(response.text, language='json')
                    
            except requests.exceptions.ConnectionError:
                st.error("无法连接到 FastAPI 服务。")
            except Exception as e:
                st.exception(e)