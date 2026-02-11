from typing import Union, List
import re
import openai
import pandas as pd
from config import DATA

from config import (
    LLM_OPENAI_API_KEY,
    LLM_OPENAI_SERVER_URL,
    LLM_MODEL_NAME,
    TFIDF_MODEL_PKL_PATH,
)

train_data = pd.read_csv(DATA, sep='\t', header=None)

client = openai.Client(
    base_url=LLM_OPENAI_SERVER_URL,
    api_key=LLM_OPENAI_API_KEY
)

# PROMPT_TEMPLATE = '''你是一个意图识别的专家，请结合待选类别和参考例子进行意图分类。
# 待选类别：{2}
#
# 历史参考例子如下：
# {1}
#
# 待识别的文本为：{0}
# 请先分析文本与类别的匹配逻辑（可详细说明），最后**必须用[[和]]包裹最终类别**（例如：[[Music-Play]]）。
# 绝对不允许在[[和]]外出现任何待选类别名称！'''
#
#
# def model_for_gpt(request_text: Union[str, List[str]]) -> List[str]:
#     classify_result: Union[str, List[str]] = []
#
#     if isinstance(request_text, str):
#         # 转换单个文本为TF-IDF向量（注意用列表包裹，因为transform需要可迭代对象）
#         tfidf_feat = tfidf.transform([request_text])  # 一个文本
#         request_text = [request_text]
#     elif isinstance(request_text, list):
#         tfidf_feat = tfidf.transform(request_text)  # 多个文本
#     else:
#         raise Exception("格式不支持")
#
#     # 遍历每个输入文本（zip将文本和索引对应，idx是当前文本在tfidf_feat中的索引）
#     for query_text, idx in zip(request_text, range(tfidf_feat.shape[0])):
#         # 动态提示词
#         # 计算当前输入文本与所有训练文本的相似度（TF-IDF向量点积）
#         # tfidf_feat[idx]是当前文本的TF-IDF向量（1行N列）
#         # train_tfidf.T是训练文本向量的转置（N行M列，M是训练样本数）
#         # 点积结果是1行M列的向量，表示当前文本与每个训练样本的相似度
#         ids = np.dot(tfidf_feat[idx], train_tfidf.T)  # 计算待推理的文本与训练哪些最相似
#         # 取相似度最高的前10个训练样本的索引
#         # toarray()[0]：将稀疏矩阵转为稠密数组（方便排序）
#         # argsort()[::-1]：按相似度从高到低排序，返回索引
#         # [:10]：取前10个最相似的样本
#         top10_index = ids.toarray()[0].argsort()[::-1][:10]
#
#         # 组织为字符串
#         # 将前10个相似样本格式化为"文本 -> 类别"的字符串（去掉类别中的"-"，方便LLM理解）
#         dynamic_top10 = ""
#         for similar_row in train_data.iloc[top10_index].iterrows():
#             # similar_row是元组：(索引, 行数据)，行数据第0列是文本，第1列是类别
#             dynamic_top10 += similar_row[1][0] + " -> " + similar_row[1][1].replace("-", "") + "\n"
#
#         response = client.chat.completions.create(
#             # 云端大模型、云端token
#             # 本地大模型，本地大模型地址
#             model=LLM_MODEL_NAME,
#             messages=[
#                 {"role": "user", "content": PROMPT_TEMPLATE.format(
#                     query_text,   # 待分类文本（{0}）
#                     dynamic_top10,  # 相似参考例子（{1}）
#                     "\n".join(list(train_data[1].unique()))  # 待选类别（{2}，用/分隔）
#                 )},
#             ],
#             temperature=0,  # 温度参数：0表示输出最确定的结果（无随机性）
#             max_tokens=256,  # 限制输出长度（类别名称很短，64足够）
#         )
#
#         classify_result.append(response.choices[0].message.content)
#
#     return classify_result

PROMPT_TEMPLATE = '''你是意图分类工具，需从【待选类别】中选择唯一匹配的类别。

【待选类别】：{1}
【待识别文本】：{0}

请先分析文本与类别的匹配逻辑（可详细说明），最后**必须用[[和]]包裹最终类别**（例如：[[Music-Play]]）。
绝对不允许在[[和]]外出现任何待选类别名称！'''


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


def model_for_gpt(request_text: Union[str, List[str]]) -> List[str]:
    classify_result: List[str] = []

    if isinstance(request_text, str):
        request_text = _preprocess_text(request_text)
        request_text = [request_text]
    elif isinstance(request_text, list):
        request_text = [_preprocess_text(text) for text in request_text]
    else:
        raise TypeError(f"不支持的输入格式：{type(request_text)}，仅支持str或list")

    # 1. 提前获取所有待选类别（用于后续匹配）
    ALL_CATEGORIES = list(train_data[1].unique())
    # 为了匹配更精准，给每个类别加个“边界”（避免部分匹配，比如“Play”匹配“Music-Play”）
    # 比如把“Music-Play”变成“ Music-Play ”（前后加空格），确保只匹配完整类别
    categories_with_boundary = [f" {cat} " for cat in ALL_CATEGORIES]

    # 2. 遍历每个文本处理
    for query_text in request_text:
        # 构造提示词（可以保留之前的极简版本，也可以用之前带例子的版本，反正最后会提取）
        prompt = f'''从以下类别中选择唯一匹配的：{','.join(ALL_CATEGORIES)}
                     每个类别及其判定标准为：{category_rules}
                     文本：{query_text}
                     最终必须在输出末尾明确写出类别名称（如Music-Play），其他内容随意。'''

        # 调用LLM（max_tokens=256可以保留，让模型完整输出分析+类别）
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256
        )

        # 3. 关键：提取输出末尾的类别名称
        raw_output = response.choices[0].message.content.strip()
        # 先清理输出中的特殊符号（如“”、换行）
        clean_output = raw_output.replace("”", "").replace("\n", " ").replace('"', "").replace("'", "")
        # 在清理后的输出前后也加空格，方便和“带边界的类别”匹配
        clean_output_with_boundary = f" {clean_output} "

        # 遍历所有类别，找最后一个出现的完整类别
        final_result = "Other"  # 默认类别
        last_index = -1  # 记录类别在输出中最后出现的位置
        for i, (cat, cat_with_bound) in enumerate(zip(ALL_CATEGORIES, categories_with_boundary)):
            # 用rfind()找“带边界的类别”在文本中的最后出现位置（从右往左找）
            # 示例：找" Music-Play "在" 用户想播放歌曲，匹配类别是Music-Play，其他类别不匹配 "中的位置
            index = clean_output_with_boundary.rfind(cat_with_bound)
            # 如果找到，且位置比之前的“最后位置”更靠后，更新结果
            if index > last_index:
                last_index = index  # 更新最后出现的位置
                final_result = cat  # 更新为当前类别

        # 4. 将提取到的类别加入结果列表
        classify_result.append(final_result)

    return classify_result


if __name__ == "__main__":
    texts = [
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

    result = model_for_gpt(texts)
    for i, text in enumerate(texts):
        print(f"文本：{text}")
        print(f"意图：{result[i]}\n")