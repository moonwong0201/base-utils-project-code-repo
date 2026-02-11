"""
TF-IDF 是 “翻译官”：把文本→数字向量（特征）；
SVC 是 “分类专家”：用数字向量学规则，最终判断文本属于哪个类别。
"""
import pandas as pd
import jieba
from joblib import dump  # 模型保存：将TF-IDF向量器和SVM模型保存为.pkl文件
from sklearn.svm import LinearSVC  # 分类模型：线性支持向量机（适合文本分类的轻量模型）
from sklearn.feature_extraction.text import TfidfVectorizer  # 特征提取：将文本转为TF-IDF向量
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('../assets/dataset/dataset.csv', sep='\t', header=None)

cn_stopwords = pd.read_csv('../assets/dataset/baidu_stopwords.txt', header=None)[0].values

train_data[0] = train_data[0].apply(lambda x: " ".join([x for x in jieba.lcut(x) if x not in cn_stopwords]))

x_train, x_test, train_labels, test_labels = train_test_split(
    train_data[0],             # 文本数据
    train_data[1],            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=train_data[1]    # 确保训练集和测试集的标签分布一致
)

# 初始化TF-IDF向量器：
# ngram_range=(1,1)：只提取单个词（unigram）的特征，不考虑词组（如“电视剧”是特征，“看电视剧”不是）
# 若设为(1,2)，则同时提取单个词和两个词的组合（如“电视剧”“看电视剧”都是特征）
tfidf = TfidfVectorizer(ngram_range=(1, 1))

# 对预处理后的文本进行拟合和转换：
# 1. fit：学习文本中的词汇表（如“播放”“电影”“查询”等），计算每个词的IDF值
# 2. transform：将文本转为TF-IDF向量（每行是一个样本的向量，列是词汇表中的词）
train_tfidf = tfidf.fit_transform(x_train)

# 初始化线性支持向量机（LinearSVC）：
# 适合文本分类的轻量模型，在高维特征（如TF-IDF向量）上表现好，训练速度快
model = LinearSVC()

# 训练模型：用TF-IDF向量（train_tfidf）和对应的类别标签（train_data[1]）拟合模型
# 模型会学习“TF-IDF向量→类别”的映射关系（如向量[0.8, 0.6, 0, 0]→"FilmTele-Play"）
model.fit(train_tfidf, train_labels)

test_tfidf = tfidf.transform(x_test)
accuracy = (model.predict(test_tfidf) == test_labels).mean() * 100
print(f"TF-IDF+SVM 测试集准确率：{accuracy:.2f}%")

dump((tfidf, model), "../assets/weights/tfidf_ml.pkl")  # pickle 二进制