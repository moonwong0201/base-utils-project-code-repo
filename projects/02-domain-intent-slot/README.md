# 领域、意图与槽位抽取服务

## 项目简介

本项目是一个基于 FastAPI 构建的多方案意图与槽位抽取服务，能够实现对用户文本的 **领域识别、意图分类、槽位实体抽取** 三大功能，支持 BERT 微调模型、Prompt 提示词、Function Call 三种抽取方式，可快速部署并提供标准化的 HTTP 接口。

## 项目结构

```
02-domain-intent-slot/
├── data/                    # 数据集
│   ├── domain.txt           # 领域标签
│   ├── intents.txt          # 意图标签
│   ├── slots.txt            # 槽位标签
│   ├── train_process.json   # 训练标注数据
│   └── test_process.json    # 测试标注数据
├── checkpoints/             # 配置参数
│   ├── model.pt             # BERT训练好的模型权重
├── training_code/           # 模型推理模块
│   ├── dataset.py           # 整理数据集
│   ├── main.py              # 训练主函数
│   └── preprocess.py        # 数据预处理
├── config.py                # 配置参数
├── bert.py                  # BERT方法
├── prompt.py                # 提示词方法
├── tools.py                 # function call方法
├── model.py                 # 模型定义
├── data_schema.py           # 请求/响应数据结构定义
├── main.py                  # FastAPI接口服务入口
└── README.md                # 项目说明
```

## 数据及模型准备

- 领域标签 29 个，意图标签 24 个，槽位标签 62 个

- 标注数据集 `data/train_process.json` 、 `data/test_process.json`，格式示例：

  ```
  [
    {
      "text": "电视台现在在播放什么大陆动漫",
      "domain": "epg",
      "intent": "QUERY",
      "slots": {
        "area": "大陆",
        "category": "动漫",
        "datetime_time": "现在"
      }
    },
    {
      "text": "张绍刚的综艺节目",
      "domain": "video",
      "intent": "QUERY",
      "slots": {
        "artist": "张绍刚",
        "category": "节目",
        "tag": "综艺"
      }
    },
    ...
  ```

- 预训练模型：

  - BERT 中文预训练模型（`bert-base-chinese`）

## BERT 模型训练

```
python training_code/main.py
```

训练完成后，模型权重会保存到 `checkpoints/model.pt`。

## 服务启动

### 开发环境启动

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 接口使用

启动服务后，访问 `http://localhost:8000/docs` 可查看自动生成的 Swagger 接口文档，支持在线调试。

### 核心接口

#### Bert 抽取接口

- 接口路径：```post /extract/bert```

- 接口描述：调用已训练好的 bert 模型进行抽取

- 响应示例：

  ```
  {
    "request_id": "string",
    "request_text": "打豆浆剩下的豆渣怎样做好吃？",
    "msg": "success",
    "data": [
      {
        "text": "打豆浆剩下的豆渣怎样做好吃？",
        "domain": "cookbook",
        "intent": "QUERY",
        "slots": {
          "ingredient": "豆渣"
        }
      }
    ]
  }
  ```

#### 提示词抽取接口

- 接口路径：``post /extract/prompt``
- 接口描述：调用大模型进行抽取
- 响应示例：同 Bert 抽取接口

#### Function Call 抽取接口

- 接口路径：```post /extract/tools```
- 接口描述：用大模型的 function call 功能进行抽取
- 响应示例：同 Bert 抽取接口

## 性能指标

#### Bert

对 Bert 抽取进行了测试，统计了三种功能的 **准确率、精确率、召回率和 F1**，结果如下表：

| 任务     | Accuracy | Precision | Recall | F1     |
| -------- | -------- | --------- | ------ | ------ |
| 领域识别 | 0.9128   | 0.9128    | 0.9128 | 0.9128 |
| 意图识别 | 0.9205   | 0.9205    | 0.9205 | 0.9205 |
| 实体抽取 | 0.9277   | 0.7557    | 0.8138 | 0.7837 |

从表中可以看到，模型整体性能良好，三种功能的准确率均达到 91% 以上，说明模型对常规文本的理解和识别能力较强，能够较好的适配基础的业务场景。而对于实体抽取，存在误判和漏判的情况，F1 得分也反映出识别准确性和完整性有较大优化空间。

自定义 12 句进行测试，结果如下：

```
{
  "request_id": "string",
  "request_text": [
    "查询下周三我的日程安排是什么",
    "帮我播放《狂飙》第15集",
    "今天天气不错，适合出门散步",
    "给我播放张敬轩的歌曲《春秋》",
    "帮我打开客厅的空调，设置温度26度",
    "帮我播放湖南卫视的《快乐大本营》最新一期",
    "收听FM93.8交通广播的实时路况",
    "查询明天上海的天气情况，是否有雨",
    "把明天早上7点的闹钟修改为7点30分",
    "播放《明朝那些事儿》的有声小说第10章",
    "帮我找到电影《流浪地球2》并播放",
    "查询从北京到上海的高铁票，明天下午出发的"
  ],
  "msg": "success",
  "data": [
    {
      "text": "查询下周三我的日程安排是什么",
      "domain": "epg",
      "intent": "QUERY",
      "slots": {
        "datetime_date": "下周三"
      }
    },
    {
      "text": "帮我播放《狂飙》第15集",
      "domain": "video",
      "intent": "QUERY",
      "slots": {
        "name": "狂飙》",
        "episode": "15"
      }
    },
    {
      "text": "今天天气不错，适合出门散步",
      "domain": "translation",
      "intent": "QUERY",
      "slots": {
        "datetime_date": "今天",
        "content": "出门散步"
      }
    },
    {
      "text": "给我播放张敬轩的歌曲《春秋》",
      "domain": "music",
      "intent": "PLAY",
      "slots": {
        "artist": "张敬轩",
        "song": "春秋"
      }
    },
    {
      "text": "帮我打开客厅的空调，设置温度26度",
      "domain": "poetry",
      "intent": "QUERY",
      "slots": {
        "category": "空",
        "content": "2",
        "name": "调"
      }
    },
    {
      "text": "帮我播放湖南卫视的《快乐大本营》最新一期",
      "domain": "epg",
      "intent": "QUERY",
      "slots": {
        "tvchannel": "湖南卫视",
        "name": "快乐大本营"
      }
    },
    {
      "text": "收听FM93.8交通广播的实时路况",
      "domain": "tvchannel",
      "intent": "LAUNCH",
      "slots": {
        "name": "的实"
      }
    },
    {
      "text": "查询明天上海的天气情况，是否有雨",
      "domain": "map",
      "intent": "QUERY",
      "slots": {
        "datetime_date": "明天",
        "location_city": "上海"
      }
    },
    {
      "text": "把明天早上7点的闹钟修改为7点30分",
      "domain": "message",
      "intent": "SEND",
      "slots": {
        "datetime_date": "明天",
        "datetime_time": "7点30"
      }
    },
    {
      "text": "播放《明朝那些事儿》的有声小说第10章",
      "domain": "novel",
      "intent": "QUERY",
      "slots": {
        "name": "明朝那些事儿",
        "category": "有声"
      }
    },
    {
      "text": "帮我找到电影《流浪地球2》并播放",
      "domain": "video",
      "intent": "QUERY",
      "slots": {
        "category": "电影",
        "name": "流浪地球2"
      }
    },
    {
      "text": "查询从北京到上海的高铁票，明天下午出发的",
      "domain": "train",
      "intent": "QUERY",
      "slots": {
        "startLoc_city": "北京",
        "endLoc_city": "上海",
        "category": "高铁",
        "startDate_date": "明天下午"
      }
    }
  ]
}
```

#### 提示词

用同样的 12 句测试，结果如下：

```
{
  "request_id": "string",
  "request_text": [
		...
  ],
  "msg": "success",
  "data": [
    {
      "text": "查询下周三我的日程安排是什么",
      "domain": "calendar",
      "intent": "QUERY",
      "slots": {
        "datetime_date": "下周三"
      }
    },
    {
      "text": "帮我播放《狂飙》第15集",
      "domain": "tvchannel",
      "intent": "PLAY",
      "slots": {
        "film": "狂飙",
        "episode": "15"
      }
    },
    {
      "text": "今天天气不错，适合出门散步",
      "domain": "weather",
      "intent": "QUERY",
      "slots": {
        "datetime_date": "今天"
      }
    },
    {
      "text": "给我播放张敬轩的歌曲《春秋》",
      "domain": "music",
      "intent": "PLAY",
      "slots": {
        "artist": "张敬轩",
        "song": "春秋"
      }
    },
    {
      "text": "帮我打开客厅的空调，设置温度26度",
      "domain": "app",
      "intent": "LAUNCH",
      "slots": {
        "location_poi": "客厅",
        "content": "空调",
        "keyword": "26度"
      }
    },
    {
      "text": "帮我播放湖南卫视的《快乐大本营》最新一期",
      "domain": "tvchannel",
      "intent": "PLAY",
      "slots": {
        "tvchannel": "湖南卫视",
        "film": "快乐大本营",
        "subfocus": "最新一期"
      }
    },
    {
      "text": "收听FM93.8交通广播的实时路况",
      "domain": "radio",
      "intent": "PLAY",
      "slots": {
        "code": "FM93.8",
        "tag": "交通广播",
        "content": "实时路况"
      }
    },
    {
      "text": "查询明天上海的天气情况，是否有雨",
      "domain": "weather",
      "intent": "QUERY",
      "slots": {
        "datetime_date": "明天",
        "location_city": "上海",
        "keyword": "是否有雨"
      }
    },
    {
      "text": "把明天早上7点的闹钟修改为7点30分",
      "domain": "app",
      "intent": "CREATE",
      "slots": {
        "datetime_date": "明天",
        "datetime_time": "7点",
        "timeDescr": "7点30分"
      }
    },
    {
      "text": "播放《明朝那些事儿》的有声小说第10章",
      "domain": "novel",
      "intent": "PLAY",
      "slots": {
        "book": "明朝那些事儿",
        "chapter": "第10章",
        "type": "有声小说"
      }
    },
    {
      "text": "帮我找到电影《流浪地球2》并播放",
      "domain": "cinemas",
      "intent": "SEARCH",
      "slots": {
        "film": "流浪地球2"
      }
    },
    {
      "text": "查询从北京到上海的高铁票，明天下午出发的",
      "domain": "train",
      "intent": "QUERY",
      "slots": {
        "startLoc_city": "北京",
        "endLoc_city": "上海",
        "datetime_date": "明天",
        "timeDescr": "下午"
      }
    }
  ]
}
```

#### Function Call

用同样的 12 句测试，结果如下：

```
{
  "request_id": "string",
  "request_text": [
		...
  ],
  "msg": "success",
  "data": [
    {
      "text": "查询下周三我的日程安排是什么",
      "domain": null,
      "intent": null,
      "slots": {}
    },
    {
      "text": "帮我播放《狂飙》第15集",
      "domain": "video",
      "intent": "PLAY",
      "slots": {
        "film": "狂飙",
        "episode": "15"
      }
    },
    {},
    {
      "text": "给我播放张敬轩的歌曲《春秋》",
      "domain": "music",
      "intent": "PLAY",
      "slots": {
        "artist": "张敬轩",
        "song": "春秋"
      }
    },
    {
      "text": "帮我打开客厅的空调，设置温度26度",
      "domain": null,
      "intent": null,
      "slots": {}
    },
    {
      "text": "帮我播放湖南卫视的《快乐大本营》最新一期",
      "domain": "tvchannel",
      "intent": "PLAY",
      "slots": {
        "tvchannel": "湖南卫视",
        "keyword": "快乐大本营"
      }
    },
    {
      "text": "收听FM93.8交通广播的实时路况",
      "domain": "radio",
      "intent": "PLAY",
      "slots": {
        "tvchannel": "FM93.8交通广播",
        "content": "实时路况"
      }
    },
    {
      "text": "查询明天上海的天气情况，是否有雨",
      "domain": "weather",
      "intent": "QUERY",
      "slots": {
        "location_city": "上海",
        "date": "明天"
      }
    },
    {
      "text": "把明天早上7点的闹钟修改为7点30分",
      "domain": "app",
      "intent": "OPEN",
      "slots": {
        "datetime_time": "7点30分"
      }
    },
    {
      "text": "播放《明朝那些事儿》的有声小说第10章",
      "domain": "novel",
      "intent": "PLAY",
      "slots": {
        "name": "明朝那些事儿",
        "episode": "10"
      }
    },
    {
      "text": "帮我找到电影《流浪地球2》并播放",
      "domain": "cinemas",
      "intent": "PLAY",
      "slots": {
        "film": "流浪地球2"
      }
    },
    {
      "text": "查询从北京到上海的高铁票，明天下午出发的",
      "domain": "train",
      "intent": "SEARCH",
      "slots": {
        "startLoc_city": "北京",
        "endLoc_city": "上海",
        "datetime_time": "明天下午"
      }
    }
  ]
}
```

可以看出，三种方法都基本能实现领域识别、意图识别和实体抽取。而提示词和 Bert 模型效果更好，且更加稳定，而 function call 方法受到模型随机性影响，表现略弱于前两者。

#### 优化

其实在未优化之前，对于槽位填充，除了准确率其余指标均在 70% 以下，损失采用的是总体损失也就是三种功能的损失之和，即：

```
loss = domain_loss + seq_loss + token_loss
```

实体抽取的损失权重和其余两个一样，这就导致模型对实体抽取功能的关注度不够，于是可以调整损失函数权重，提升实体抽取损失权重，为：

```
loss = 0.2 * domain_loss + 0.2 * seq_loss + 1.0 * token_loss
```

并且增大 Dropout，效果有了显著提升。

