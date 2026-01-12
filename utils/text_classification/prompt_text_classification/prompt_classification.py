import pandas as pd
from openai import OpenAI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
# dataset = pd.read_csv(
#     "/Users/wangyingyue/materials/大模型学习资料——八斗/第一周：课程介绍及大模型基础/Week01/Week01/dataset.csv",
#     sep="\t",
#     header=None
# )

# 云端大模型
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-078ae61448344f53b3cb03bcc85ff7cd",

    # 大模型厂商的地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 构建提示词
def build_prompt(text):
    prompt_template = f"""你是一个专业文本分析专家，请帮我对如下的文本进行分类：
{text}

可以参考的样本如下（假如已有的训练集）：
查询北京飞桂林的飞机是否已经起飞了	Travel-Query
从这里怎么回家	Travel-Query
随便播放一首专辑阁楼里的佛里的歌	Music-Play
给看一下墓王之王嘛	FilmTele-Play
我想看挑战两把s686打突变团竞的游戏视频	Video-Play
我想看和平精英上战神必备技巧的游戏视频	Video-Play

你只能从如下的类别选择：['FilmTele-Play', 'Video-Play', 'Music-Play', 'Radio-Listen',
       'Alarm-Update', 'Weather-Query', 'Travel-Query',
       'HomeAppliance-Control', 'Calendar-Query', 'TVProgram-Play',
       'Audio-Play', 'Other']

只需要输出结果，不需要额外的解释。
"""
    return prompt_template


# 调用大模型进行分类
def classify_text(text):
    prompt = build_prompt(text)
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=20
    )
    classification_result = completion.choices[0].message.content.strip()
    return classification_result


if __name__ == "__main__":
    test_cases = [
        '查询下周三我的日程安排是什么',
        '帮我播放《狂飙》第15集',
        '今天天气不错，适合出门散步',
        '给我播放张敬轩的歌曲《春秋》',
        '帮我打开客厅的空调，设置温度26度',
        '帮我播放湖南卫视的《快乐大本营》最新一期',
        '收听FM93.8交通广播的实时路况',
        '查询明天上海的天气情况，是否有雨',
        '把明天早上7点的闹钟修改为7点30分',
        '播放《明朝那些事儿》的有声小说第10章',
        '帮我找到电影《流浪地球2》并播放',
        '查询从北京到上海的高铁票，明天下午出发的'
    ]
    for text in test_cases:
        result = classify_text(text)
        print(f"待分类文本：{text}")
        print(f"分类结果：{result}")

