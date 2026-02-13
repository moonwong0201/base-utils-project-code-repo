import json
import requests
# FastAPI 官方提供的测试客户端，核心作用是模拟浏览器 / Postman 发送 HTTP 请求，无需启动 uvicorn 服务
from fastapi.testclient import TestClient
from main import app

# 创建一个绑定到项目 app 的测试客户端，后续所有 client.post/get/delete 都会直接调用 app 中的路由函数，无需启动独立服务
client = TestClient(app)

"""
使用 pytest 框架和 FastAPI 自带的 TestClient 编写的API 集成测试脚本。
它的核心目的是自动化验证 RAG 系统的 API 接口是否按照预期工作
"""

def test_knowledge_base():
    """
    验证了 POST /create, GET /retrieve, DELETE /delete 整个 CRUD (创建、读取、更新、删除) 生命周期
    确保了这些操作之间的逻辑一致性
    不仅检查了 “成功” 的路径，还检查了 “失败” 的路径，让测试更健壮。
    """
    # 步骤 1: 创建一个新的知识库
    new_data = {"category": "测试类别", "title": "测试标题"}
    response = client.post("/v1/knowledge_base/", json=new_data)  # 查询知识库
    # 断言：检查响应状态码是否为 200 (OK)
    assert response.status_code == 200
    # 断言：检查 API 返回的业务状态码是否为 200 (表示成功)
    assert response.json()["response_code"] == 200

    # 从创建成功的响应中提取新创建的知识库 ID
    knowledge_id = response.json()["knowledge_id"]

    # 步骤 2: 查询刚刚创建的知识库
    response = client.get(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=666")
    # 断言：检查查询是否成功
    assert response.status_code == 200
    # 断言：检查返回的知识库标题是否与创建时一致
    assert response.json()["title"] == "测试标题"
    # 步骤 3: 删除刚刚创建的知识库
    response = client.delete(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=666")
    # 断言：检查删除是否成功
    assert response.status_code == 200
    assert response.json()["response_msg"] == "知识库删除成功"

    # 步骤 4: 验证删除操作 (尝试删除一个已不存在的知识库)
    response = client.delete(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=666")
    # 断言：检查系统是否正确处理了“删除不存在资源”的情况
    assert response.status_code == 200
    assert response.json()["response_msg"] == "知识库不存在"


def test_document():
    # 步骤 1: 尝试查询一个不存在的文档
    response = client.get(f"/v1/document?document_id=666&token=666")
    # 断言：检查系统是否正确处理了“查询不存在资源”的情况
    assert response.status_code == 200
    assert response.json()["response_msg"] == "文档不存在"
    # 步骤 2: 尝试删除一个不存在的文档
    response = client.delete(f"/v1/document?document_id=666&token=666")
    # 断言：检查系统是否正确处理了“删除不存在资源”的情况
    assert response.status_code == 200
    assert response.json()["response_msg"] == "文档不存在"


"""
当执行 pytest test/test_api.py -v 时：
pytest 找到 test_api.py 文件（符合 test_*.py 规则）；
扫描文件内所有以 test_ 开头的函数：test_knowledge_base、test_document；
按顺序执行这两个函数：
执行 test_knowledge_base()：依次跑 “创建→查询→删除→验证删除”，所有 assert 都通过 → 标记为 PASSED；
执行 test_document()：跑 “查询不存在文档→删除不存在文档”，所有 assert 都通过 → 标记为 PASSED；
输出汇总结果（2 passed），结束运行。
"""
