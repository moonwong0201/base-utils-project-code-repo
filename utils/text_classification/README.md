# 文本分类多方法实现（Text Classification with Multiple Approaches）
> 一个包含多种经典/主流文本分类方法的实现仓库，涵盖传统机器学习、深度学习（CNN/RNN/Transformer）等方案，支持直接运行和二次开发。

## 一、 项目概述（Project Overview）
### 1.1 项目简介
本仓库旨在整理和实现多种文本分类的核心方法，从传统的基于统计特征的机器学习模型，到现代的基于深度语义的预训练语言模型，提供一套完整、可复现的文本分类解决方案。
- 适合文本分类入门者学习不同方法的差异与实现思路
- 可为实际业务场景提供快速的模型选型参考和代码复用
- 所有模型均基于公开数据集验证（可替换为自定义数据集）

### 1.2 支持的分类方法
目前已实现以下文本分类方法，持续更新中：
| 方法类型 | 具体模型/算法 | 核心依赖 | 适用场景 |
|----------|---------------|----------|----------|
| 传统机器学习 | Naive Bayes（朴素贝叶斯） | scikit-learn | 简单场景、数据量较小 |
| 传统机器学习 | SVM（支持向量机） | scikit-learn | 文本分类基准、中等数据量 |
| 传统机器学习 | Logistic Regression（逻辑回归） | scikit-learn | 易解释、需要快速落地 |
| 深度学习（基础） | CNN（卷积神经网络） | PyTorch/TensorFlow | 捕捉局部文本特征、短文本 |
| 深度学习（基础） | LSTM（长短期记忆网络） | PyTorch/TensorFlow | 捕捉文本序列依赖、长文本 |
| 预训练语言模型 | BERT（Bidirectional Encoder Representations from Transformers） | Hugging Face Transformers | 高精度需求、充足算力、各类文本场景 |
| 预训练语言模型 | RoBERTa（优化版BERT） | Hugging Face Transformers | 比BERT更高的精度、充足算力 |
