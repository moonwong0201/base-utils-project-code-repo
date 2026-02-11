# 用户意图识别系统

## 项目简介

本项目是一套用户意图识别方案，支持正则规则、TF-IDF、BERT、大语言模型（LLM）四种分类方式，可直接集成到车载语音助手、智能客服系统中。

- 正则规则（极速）、TF-IDF+SVM（稳定）、BERT（高精度）、LLM（泛化性强）
- 支持 12100 条标注数据集，覆盖各种场景

## 项目结构

```
01-intent-classify/
├── assets/                  
│   ├── label2id.json        # BERT标签映射表（训练自动生成）
│   ├── dataset/             # 数据集
│   │   ├── dataset.csv      # 意图标注数据（文本\t标签）
│   │   └── baidu_stopwords.txt # 停用词表
│   └── weights/             # 训练好的模型权重
│       ├── bert.pt          # BERT微调权重
│       └── tfidf_ml.pkl     # TF-IDF+SVM模型权重
├── config.py                # 配置参数
├── data_schema.py           # 请求/响应数据结构定义
├── logger.py                # 日志配置
├── main.py                  # FastAPI接口服务入口
├── model/                   # 模型推理模块
│   ├── bert.py              # BERT推理逻辑
│   ├── prompt.py            # LLM推理逻辑
│   ├── regex_rule.py        # 正则规则推理
│   └── tfidf_ml.py          # TF-IDF+SVM推理
├── training_code/           # 模型训练/测试脚本
│   ├── train_regex.py       # 正则测试脚本
│   ├── train_tfidf.py       # TF-IDF+SVM 训练/测试脚本
│   ├── train_bert.py        # BERT 训练/测试脚本
│   └── train_prompt.py      # prompt 测试脚本
└── README.md                # 项目说明
```

## 数据及模型准备

- 标注数据集 `assets/dataset/dataset.csv`，格式示例：

  ```
  还有双鸭山到淮阴的汽车票吗13号的    Travel-Query
  随便播放一首专辑阁楼里的佛里的歌    Music-Play
  给看一下墓王之王嘛    FilmTele-Play
  ```

- 停用词表 `assets/dataset/baidu_stopwords.txt`。

- 预训练模型：
  - BERT 中文预训练模型（`bert-base-chinese`），放入 `assets/models/bert-base-chinese/`

## 模型训练

### 1. TF-IDF+SVM 训练

```
python train_tfidf.py
```

训练完成后，模型权重会保存到 `assets/weights/tfidf_ml.pkl`。

### 2. BERT 训练

```
python train_bert.py
```

- 训练完成后，模型权重保存到 `assets/weights/bert.pt`；
- 标签映射表自动生成到 `assets/config/label2id.json`，保证训练 / 推理标签顺序一致；
- 训练过程中会输出每轮准确率，最终保存最优模型。

## 服务启动

### 开发环境启动

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 接口使用

启动服务后，访问 `http://localhost:8000/docs` 可查看自动生成的 Swagger 接口文档，支持在线调试。

### 核心接口

#### 健康检查接口

- **接口路径**：`GET /health`

- **接口描述**：轻量检测服务是否存活

- **请求参数**：无

- **响应示例**（200 OK）：

  ```
  {
    "status": "healthy",
    "service": "intent-classify",
    "version": "1.0",
    "timestamp": "2026-02-11 17:30:00",
    "message": "服务正常运行"
  }
  ```

#### 意图分类接口

##### 正则分类接口

- **接口路径**：`POST /v1/text-cls/regex`

- **接口描述**：通过正则表达式进行意图分类

- **请求体**（Content-Type: application/json）：

  ```
  {
    "request_id": "test_001",
    "request_text": ["帮我播放《狂飙》第15集"]
  }
  ```

- **响应示例**（200 OK）：

  ```
  {
    "request_id": "test_001",
    "request_text": ["帮我播放《狂飙》第15集"],
    "classify_result": "FilmTele-Play",
    "classify_time": 0.001,
    "error_msg": "ok"
  }
  ```

##### TF-IDF 分类接口

- **接口路径**：`POST /v1/text-cls/tfidf`
- **接口描述**：通过 TF-IDF+SVM 进行意图分类
- **请求体**：同正则接口
- **响应示例**：同正则接口

##### BERT 分类接口

- **接口路径**：`POST /v1/text-cls/bert`
- **接口描述**：通过 BERT 进行意图分类
- **请求体**：同正则接口
- **响应示例**：同正则接口

##### GPT 分类接口

- **接口路径**：`POST /v1/text-cls/gpt`
- **接口描述**：通过大语言模型进行意图分类
- **请求体**：同正则接口
- **响应示例**：同正则接口

## 性能指标

| 模型类型 | 测试集准确率 | 自定义测试集准确率 | 适用场景               |
| :------: | :----------: | :----------------: | ---------------------- |
| 正则规则 |      /       |       66.67%       | 关键词明确的简单意图   |
|  TF-IDF  |   90.33%+    |       91.67%       | 通用场景、资源受限环境 |
|   BERT   |   94.83%+    |        100%        | 高精度要求的核心场景   |
|   LLM    |      /       |       83.33%       | 泛化性要求高的场景     |

在意图识别场景中，BERT 和 TF-IDF 是最优的方案。
