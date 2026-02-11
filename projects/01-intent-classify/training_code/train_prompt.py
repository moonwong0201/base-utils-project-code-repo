from typing import Union, List
import re
import openai
import pandas as pd
import numpy as np
import jieba
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time  # 新增：统计耗时

from config import (
    LLM_OPENAI_API_KEY,
    LLM_OPENAI_SERVER_URL,
    LLM_MODEL_NAME,
    TFIDF_MODEL_PKL_PATH,
)

# --------------------------
# 1. 加载数据+拆分测试集（保留你的逻辑）
# --------------------------
train_data = pd.read_csv('../assets/dataset/dataset.csv', sep='\t', header=None)

client = openai.Client(
    base_url=LLM_OPENAI_SERVER_URL,
    api_key=LLM_OPENAI_API_KEY
)

x_train, x_test, train_labels, test_labels = train_test_split(
    train_data[0],  # 文本数据
    train_data[1],  # 对应的数字标签
    test_size=0.2,  # 测试集比例为20%
    stratify=train_data[1],  # 确保训练集和测试集的标签分布一致
    random_state=42  # 固定种子，结果可复现
)

# --------------------------
# 2. 核心配置（保留你的类别规则）
# --------------------------
category_rules = """
### 意图类别及判定标准（必须严格遵守）
1. Travel-Query：与出行相关（车票/机票/导航/路线/行程/回家/打车/路况/停车等）；
2. Music-Play：与音乐播放相关（播放歌曲/专辑/听歌/点歌/音乐/单曲等）；
3. FilmTele-Play：与影视播放相关（播放电视剧/电影/追剧/看剧/影片等）；
4. Video-Play：与非影视类视频播放相关（游戏视频/短视频/直播/录屏/刷视频等）；
5. Radio-Listen：与听广播相关（听电台/广播/调频/收音机等）；
6. HomeAppliance-Control：与智能家居控制相关（开关空调/灯光/调温/窗帘/冰箱等）；
7. Weather-Query：与天气查询相关（天气/温度/下雨/预报/防晒/雨伞等）；
8. Alarm-Update：与闹钟相关（设置/修改/关闭闹钟/提醒/定时等）；
9. Calendar-Query：与日历查询相关（查日期/节假日/行程安排/纪念日等）；
10. TVProgram-Play：与电视节目播放相关（播放电视节目/卫视/综艺频道等）；
11. Audio-Play：与音频播放相关（播放有声书/播客/相声/评书等非音乐音频）；
12. Other：无法匹配以上任何类别的文本。
"""


# --------------------------
# 3. 文本预处理（保留你的逻辑）
# --------------------------
def _preprocess_text(text: str) -> str:
    """文本预处理：清洗脏数据，提升BERT语义理解效果"""
    if not isinstance(text, str):
        return ""
    # 1. 转小写 + 去首尾空格
    text = text.strip().lower()
    # 2. 去除特殊符号（保留中文/字母/数字/空格，避免破坏语义）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
    # 3. 去除多余空格
    text = re.sub(r"\s+", " ", text)
    # 4. 空文本兜底
    return text if text else "无内容"


# --------------------------
# 4. 批量调用LLM（核心优化）
# --------------------------
def model_for_gpt_batch(request_texts: List[str]) -> List[str]:
    processed_texts = [_preprocess_text(text) for text in request_texts]

    # 固定类别，绝对不乱
    CATEGORIES = [
        'Travel-Query', 'Music-Play', 'FilmTele-Play', 'Video-Play',
        'Radio-Listen', 'HomeAppliance-Control', 'Weather-Query',
        'Alarm-Update', 'Calendar-Query', 'TVProgram-Play', 'Audio-Play', 'Other'
    ]

    # 超级干净、绝对不会误解的 prompt
    prompt = "请对下面每一句话做意图分类，只输出类别，不要编号，不要解释，不要多余内容。\n"
    prompt += "允许的类别：" + ", ".join(CATEGORIES) + "\n\n"
    prompt += "句子：\n"

    for i, t in enumerate(processed_texts):
        prompt += f"- {t}\n"

    prompt += "\n请按上面句子的顺序，每行只输出一个类别：\n"

    # 调用
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=len(processed_texts) * 15,
    )

    # 拿原始输出
    raw = response.choices[0].message.content.strip()
    lines = raw.split("\n")

    # 解析：每行只保留合法类别
    result = []
    for line in lines:
        line = line.strip()
        # 在这一行里找匹配的类别
        matched = "Other"
        for cat in CATEGORIES:
            if cat == line:  # 必须完全相等，防止瞎匹配
                matched = cat
                break
        result.append(matched)

    # 防止输出行数不够
    while len(result) < len(request_texts):
        result.append("Other")

    return result[:len(request_texts)]

# --------------------------
# 5. 运行测试+计算准确率
# --------------------------
if __name__ == "__main__":
    # 转换测试集为列表（保持顺序）
    x_test_list = list(x_test)
    test_labels_list = list(test_labels)

    # 批量调用LLM
    start_total = time.time()
    result = model_for_gpt_batch(x_test_list)
    total_time = time.time() - start_total

    # 计算准确率
    accuracy = (np.array(result) == np.array(test_labels_list)).mean() * 100

    # 输出结果
    print(f"Prompt 测试集准确率：{accuracy:.2f}%")
    print(f"总耗时（含预处理+解析）：{total_time:.2f}秒")
    print(f"平均每条耗时：{total_time / len(x_test_list) * 1000:.2f}毫秒")