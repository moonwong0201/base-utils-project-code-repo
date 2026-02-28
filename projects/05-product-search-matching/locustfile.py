# from locust import HttpUser, task, between
# import random
#
#
# class SearchUser(HttpUser):
#     # 用户思考时间：每次请求间隔 1-3 秒
#     wait_time = between(1, 3)
#
#     # 测试数据（模拟真实搜索词）
#     test_words = ["手机", "连衣裙", "运动鞋", "耳机", "背包", "手表", "键盘", "鼠标"]
#
#     @task
#     def search_text(self):
#         """测试文本搜文本接口"""
#         self.client.post(
#             "/product/search",
#             json={
#                 "search_type": "text2text",
#                 "query_text": random.choice(self.test_words),
#                 "top_k": 10
#             }
#         )


from locust import HttpUser, task, between
import random
import string


class SearchUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 缩短间隔，加大压力

    @task
    def search_text(self):
        # 生成随机 6 位小写字母，几乎不可能重复
        random_word = ''.join(random.choices(string.ascii_lowercase, k=6))

        self.client.post(
            "/product/search",
            json={
                "search_type": "text2text",
                "query_text": random_word,  # 每次都不一样，强制走模型
                "top_k": 10
            }
        )
