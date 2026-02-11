from typing import Union, List

import jieba
import pandas as pd
from joblib import load
from config import TFIDF_MODEL_PKL_PATH
import re


# 加载模型
# tfidf负责文本转特征，model负责特征转类别
def _load_tfidf_model():
    try:
        tfidf, model = load(TFIDF_MODEL_PKL_PATH)
        return tfidf, model
    except FileNotFoundError:
        raise Exception(f"模型文件不存在，请检查路径：{TFIDF_MODEL_PKL_PATH}")
    except Exception as e:
        raise Exception(f"模型加载失败：{str(e)}")


tfidf, model = _load_tfidf_model()

# 停用词
cn_stopwords = pd.read_csv('http://mirror.coggle.club/stopwords/baidu_stopwords.txt', header=None)[0].values


def _preprocess_text(text: str) -> str:
    """文本预处理：清洗脏数据，提升分词效果"""
    if not isinstance(text, str):
        return ""
    # 1. 转小写 + 去首尾空格
    text = text.strip().lower()
    # 2. 去除特殊符号（只保留中文、字母、数字）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)
    # 3. 去除多余空格
    text = re.sub(r"\s+", "", text)
    return text


def model_for_tfidf(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    if request_text is None:
        raise ValueError("请求文本不能为空")

    if isinstance(request_text, str):
        request_text = _preprocess_text(request_text)
        query_words = " ".join([x for x in jieba.lcut(request_text) if x not in cn_stopwords])
        # 文本转特征：用tfidf.transform()将处理后的文本转为TF-IDF向量
        # 预测类别：用model.predict()得到分类结果（返回的是列表，所以用list()包装）
        result = model.predict(tfidf.transform(query_words))
        classify_result = list(result)
    elif isinstance(request_text, list):
        query_words = []
        for text in request_text:
            request_text = _preprocess_text(text)
            query_words.append(
                " ".join([x for x in jieba.lcut(request_text) if x not in cn_stopwords])
            )
        result = model.predict(tfidf.transform(query_words))
        classify_result = list(result)
    else:
        raise Exception("格式不支持")

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

    result = model_for_tfidf(texts)
    for i, text in enumerate(texts):
        print(f"文本：{text}")
        print(f"意图：{result[i]}\n")
