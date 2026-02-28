import requests

url = "http://localhost:8000/health"
headers = {
    'accept': 'application/json'
}

response = requests.get(url, headers=headers)

print("\n####")
print(url)
print(response.text)
# 打印HTTP状态码（200表示成功，500表示服务异常）
print(f"Status Code: {response.status_code}")

