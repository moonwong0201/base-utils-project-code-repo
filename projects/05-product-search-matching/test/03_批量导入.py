import glob
import pandas as pd
import requests
import json

train = pd.read_csv("/train.csv", sep="\t")

for row in train.iloc[50:250].iterrows():
    # 后端创建商品的接口 URL
    url = "http://127.0.0.1:8000/product"
    # 构造要上传的图片文件：
    files = {
        'image': (
            row[1].path,  # 图片文件名（从 CSV 的 path 列获取）
            open(
                "/image/"
                + row[1].path, 'rb'
            ),  # 打开图片文件
            'image/jpeg'  # MIME 类型：声明为 JPEG 图片
        )
    }
    # 构造商品标题数据（从 CSV 的 title 列获取）
    data = {
        'title': row[1].title  # 商品标题，对应后端接口的 title 参数
    }
    response = requests.post(url, files=files, data=data)
    print("\n####")
    print(url)
    print(response.text)
    print(f"Status Code: {response.status_code}")
