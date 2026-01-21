import json
import pdfplumber
from openai import OpenAI


# 调用通义千问Qwen模型函数：基于OpenAI兼容模式，传入提示词返回模型回答，统一处理换行符
def ask_qwen(prompt):
    client = OpenAI(
        api_key="sk-078ae61448344f53b3cb03bcc85ff7cd",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云Qwen兼容OpenAI的API地址
    )

    response = client.chat.completions.create(
        model="qwen-turbo",  # 指定调用的Qwen轻量版模型
        messages=[{"role": "user", "content": prompt}],  # 构造用户请求消息
        temperature=0.01,  # 低温度保证回答的稳定性和确定性
        max_tokens=1000,  # 限制模型生成回答的最大令牌数
        timeout=15  # 设置请求超时时间，避免卡死
    )
    answer = response.choices[0].message.content.strip()
    return " ".join(answer.splitlines())  # 去除回答中的换行符，统一为单行


# 加载PDF文档和BGE+BM25融合重排序后的检索结果（含每个问题匹配的最优PDF页码）
pdf = pdfplumber.open(
    "/Users/wangyingyue/materials/大模型学习资料——八斗/第六周：RAG工程化实现/Week06/Week06/汽车知识手册.pdf")
bge_bm25 = json.load(open('submit_fusion_bge+bm25_rerank_retrieval.json'))

fusion_result = []
pdf_content = []
# 解析PDF所有页面，按页存储页码标识和文本内容，空文本做兜底处理
for idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(idx + 1),
        'content': pdf.pages[idx].extract_text() or ''
    })

# 遍历每个重排序后的问题，拼接参考资料并调用大模型生成回答
for idx, q in enumerate(bge_bm25):
    question = q['question']
    page_str = q['reference']
    page = int(page_str.split("_")[1])  # 解析最优匹配的PDF页码
    content = pdf_content[page - 1]['content']  # 获取对应页码的文本内容
    reference_content = content.replace('\n', ' ') + f" 上述内容在第 {page} 页"  # 整理参考资料，标注页码

    print("【用户提问】\n" + question)
    # print("【参考资料】\n" + content)

    # 构造大模型提示词：指定汽车专家角色，明确回答规则，拼接参考资料和问题
    prompt = '''你是一个汽车专家，你擅长编写和回答汽车相关的用户提问，帮我结合给定的资料，回答下面的问题。
    如果问题无法从资料中获得，或无法从资料中进行回答，请回答：无法回答。如果提问不符合逻辑，请回答：无法回答。
    如果问题可以从资料中获得，则请逐步回答。

    资料：{0}


    问题：{1}
    '''.format(reference_content, question)

    # 调用大模型，最多重试5次，处理异常情况
    answer = "无法"
    for _ in range(5):
        try:
            answer = ask_qwen(prompt)
            if answer:
                break
        except Exception as e:
            print(f"调用失败，错误原因：{str(e)}")

    # 统一处理模型返回的"无法回答"相关结果
    if "无法" in answer:
        answer = "结合给定的资料，无法回答问题。"

    q['answer'] = answer  # 为问题添加模型生成的回答
    print("【模型回答】\n" + q['answer'])

    fusion_result.append(q)

# 保存最终结果：问题+最优匹配页码+大模型回答
with open('submit_fusion_bge+bm25_rerank_retrieval_qwen.json', 'w', encoding='utf-8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)