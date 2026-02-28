import requests

# 添加商品信息
url = "http://localhost:8000/product"
file = {
    'image': (
        'pokemon.jpeg',
        open('/pokemon.jpeg', 'rb'),
        'image/jpeg'
    )
}
data = {
    'title': '正常皮卡丘'
}
response = requests.post(url, files=file, data=data)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


# 展示商品列表
url = "http://localhost:8000/product/list"
headers = {
    'accept': 'application/json'
}
response = requests.get(url, headers=headers)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


# 查询单个商品
url = "http://localhost:8000/product/1"
response = requests.get(url)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")


# 修改单个商品信息
url = "http://localhost:8000/product/1/title"
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded'
}
data = {
    "title": "修改皮卡丘"
}
response = requests.patch(url, headers=headers, data=data)
print("\n####")
print(url)
print(f"Status Code: {response.status_code}")
print(response.text)


url = "http://localhost:8000/product/5/image"
with open("/Users/wangyingyue/materials/FastAPI/code/www/img/hinscheung.jpg", "rb") as f:
    files = {
        'image': ('hinscheung.jpg', f)
    }
    response = requests.patch(url, files=files)
    print("\n####")
    print(url)
    print(f"Status Code: {response.status_code}")
    print(response.text)


# 删除指定商品
url = "http://localhost:8000/product/4"
response = requests.delete(url)
print("\n####")
print(url)
print(response.text)
print(f"Status Code: {response.status_code}")
