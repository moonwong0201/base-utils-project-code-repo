# -*- coding: utf-8 -*-
"""
项目名称：中文文本多分类（智能指令分类）
功能描述：使用TF-IDF提取文本特征，集成多种机器学习模型进行分类，包含交叉验证评估和测试集预测
"""

import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# 加载数据集（注：使用者需修改为自己的数据集路径）
dataset = pd.read_csv(
    "dataset.csv",
    sep="\t",
    header=None
)

# 中文文本分词处理（适配Sklearn的文本特征提取要求，用空格拼接分词结果）
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# TF-IDF文本特征提取（将文本转换为高维稀疏数值特征）
vector = TfidfVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

# 标签编码（将文本标签转换为模型可处理的数值标签）
lbl = LabelEncoder()
lbl.fit(dataset[1].values)
labels = lbl.transform(dataset[1].values)

# 获取原始类别名称列表
target_names = lbl.classes_.tolist()

# 构建待评估的机器学习模型字典（使用默认参数，保证稳定性和可运行性）
models_dict = {
    "knn": KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "naive_bayes": MultinomialNB(),
    "linear_svm": LinearSVC(random_state=42, max_iter=10000)
}

# 交叉验证评估各模型性能，输出分类报告
for model_name, model in models_dict.items():
    y_pred = cross_val_predict(model, input_feature, labels, cv=5)
    print(f"{model_name} 交叉验证结果：")
    print(f"平均准确率: {accuracy_score(labels, y_pred):.4f}")
    report = classification_report(labels, y_pred, target_names=target_names, zero_division=0)
    print(report)

def train_model(X, y, models_dict):
    """
    训练所有模型并返回训练好的模型字典
    :param X: 输入特征矩阵
    :param y: 标签数组
    :param models_dict: 待训练的模型字典
    :return: 训练完成的模型字典
    """
    train_models = {}
    for model_name, model in models_dict.items():
        print(f"正在训练 {model_name} 模型...")
        model.fit(X, y)
        train_models[model_name] = model
        print(f"{model_name} 模型训练完成！")
    return train_models

def predict(test_cases, models_dict):
    """
    使用训练好的模型对测试文本进行分类预测
    :param test_cases: 测试文本列表（单个文本也可自动转为列表）
    :param models_dict: 训练完成的模型字典
    :return: 无返回值，直接打印预测结果
    """
    if not isinstance(test_cases, list):
        test_cases = [test_cases]

    for text in test_cases:
        print(f"\n文本：{text}")
        text = " ".join(jieba.lcut(text))
        text_feature = vector.transform([text])

        for model_name, model in models_dict.items():
            pred_encoded = model.predict(text_feature)
            label = lbl.inverse_transform(pred_encoded)
            print(f"{model_name}预测结果：{label[0]}")

if __name__ == "__main__":
    # 训练所有模型
    trained_models = train_model(input_feature, labels, models_dict)
    # 定义测试用例（智能指令类文本）
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
    # 执行预测并输出结果
    predict(test_cases, trained_models)
