import pandas as pd
import jieba
from joblib import dump  # 模型保存：将TF-IDF向量器和SVM模型保存为.pkl文件
from sklearn.svm import LinearSVC  # 分类模型：线性支持向量机（适合文本分类的轻量模型）
from sklearn.feature_extraction.text import TfidfVectorizer  # 特征提取：将文本转为TF-IDF向量
from sklearn.model_selection import train_test_split
from config import REGEX_RULE
import re

train_data = pd.read_csv('../assets/dataset/dataset.csv', sep='\t', header=None)

cn_stopwords = pd.read_csv('../assets/dataset/baidu_stopwords.txt', header=None)[0].values


x_train, x_test, train_labels, test_labels = train_test_split(
    train_data[0],             # 文本数据
    train_data[1],            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=train_data[1]    # 确保训练集和测试集的标签分布一致
)

REGEX_RULE_COMPILED = {}
# 1. 取出该类别下的所有关键词/模式列表（比如REGEX_RULE["售后问题"] = ["退款", "退货", "换货"]）
# 2. 用"|"拼接成一个正则表达式字符串（比如"退款|退货|换货"）
# 3. 用re.compile()编译成正则对象（预编译可以提高后续匹配效率）
for category in REGEX_RULE.keys():
    REGEX_RULE_COMPILED[category] = re.compile("|".join(REGEX_RULE[category]))

classify_result = []

for text in x_test:
    is_classified = False  # 标记该文本是否已分类
    for category in REGEX_RULE_COMPILED.keys():
        if REGEX_RULE_COMPILED[category].findall(text):
            classify_result.append(category)
            is_classified = True
            break
    if not is_classified:
        classify_result.append("Other")

accuracy = (classify_result == test_labels).mean() * 100
print(f"REGEX 测试集准确率：{accuracy:.2f}%")
