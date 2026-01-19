# GPT2对话微调

## 数据集说明

- 仅支持 ```User: 问题\n AI: 回复``` 的单行配对格式
- 每组对话必须都要有 1 个 User 提问 + 1 个 AI 回复，不能缺失
- 一共 225 组对话

数据示例：

```
User: hi, how are you?
AI: i am doing well, thank you. how about you?

User: what is the weather like today?
AI: please check a local weather app for the latest conditions.

User: can you recommend a good book?
AI: sure! To Kill a Mockingbird by Harper Lee is a classic worth reading.
```

### 分离方法

#### 项目简介

基于 GPT2 模型的英文对话微调项目，采用「分离式数据处理」方法（拆分 User 输入与 AI 回复分别处理），支持从自定义 `chat.txt` 数据集加载数据，完成模型微调后，通过 Beam Search 策略生成对话回复。

#### 核心功能

- 采用「分离式数据处理」，拆分 **User** 输入与 **AI** 回复，分别进行 tokenizer 与填充；
- Beam Search 生成策略，保证回复的严谨性与连贯性；
- 权重自动保存与加载，避免重复训练

#### 训练效果

- 训练过程与生成效果显示 **损失表现良好，但生成效果不理想**；
- 回复极易容易出现冗余的内容，比如某个词不停循环；
- 对陌生的问题泛化能力较弱，回复内容不连贯

训练损失变化：

```
Epoch: 01/100, Average Loss: 7.485942
Epoch: 11/100, Average Loss: 3.237190
Epoch: 21/100, Average Loss: 2.302738
Epoch: 31/100, Average Loss: 1.903099
Epoch: 41/100, Average Loss: 1.644095
Epoch: 51/100, Average Loss: 1.415153
Epoch: 61/100, Average Loss: 1.141347
Epoch: 71/100, Average Loss: 0.961676
Epoch: 81/100, Average Loss: 0.556365
Epoch: 91/100, Average Loss: 0.392130
```

预测结果：

```
========== 测试生成结果 ==========
测试 1:
User: what is the weather like today?
AI:  the weather like today? warm or shower clothesfree and shower capstan hair conditioner hair conditioner prevents hair damage reducing 2F moisture storage using water hydration� using water sip more easily yourzz in minutes hair damage 250

测试 2:
User: hi, how are you?
AI: w are you? how great how wonderfulhow awesomehow awesome awesome awesome awesome awesomehow awesomehow awesome awesomeally awesomeally awesomeallyhow awesomeally awesomeally awesomeallyhow awesomeallyhow awesomeallyhow awesomeALLY awesomeally

测试 3:
User: can you recommend a good book?
AI:  recommend a good book? classic andawe classic andaworlde classicism iswarmingtherapyhire therapy loving therapy loving therapy loving therapy sessions therapy alone confinementisinwater wandlet bathingashandedreat paralysed person control wand
```

#### 结果推测

- 损失好但效果差的核心原因在于模型拟合了数据，但是没有学习到对话的逻辑；
- 交叉熵损失计算的是模型对单个token的预测误差，损失低只能够说明模型记住了训练数据中的token排列，也就是死记硬背住了顺序，不代表理解了对话的语义和逻辑；
- 分离式数据处理具有很大的局限性，就是拆分 User 输入和 AI 回复分别训练，模型无法学习到提问和回复的关联，只能单独学习每一个句子；
- 基础 GPT2 模型是有短板的，因为 GPT2 是文本续写模型，而非专用对话模型，缺乏对话交互的预训练逻辑

后续的优化就针对于分离式数据处理，将 User 输入和 AI 输出融合为完整的上下文序列。



### 融合方法

#### 项目简介

基于 GPT2 模型的英文对话微调项目，采用「融合式数据处理」方法（将 User 输入与 AI 回复融合成一个序列），支持从自定义 `chat.txt` 数据集加载数据，完成模型微调后，通过 Transformer 内置 Beam Search 策略生成对话回复。

#### 核心功能

- 采用「融合式数据处理」，将 **User** 输入与 **AI** 回复合并为一个序列，

  即 ```User tokens + SEP + AI tokens + <eos>``` 进行 tokenizer 与填充；

- 自定义分隔符区分问答的边界；

- 使用Transformer 内置的 **generate**，支持Beam Search；

- 权重自动保存与加载，避免重复训练

#### 主要优化

- 从分离式到融合式序列建模；
- 将手写 Beam Search 改为使用 Transformer 的 generate，其中内置了 Beam Search；

#### 训练效果

模型已经能够稳定生成符合对话语义的回复。

训练损失变化：

```
Epoch 000 | loss 2.4032 
Epoch 010 | loss 0.1315 
Epoch 020 | loss 0.0313 
Epoch 030 | loss 0.0173 
Epoch 040 | loss 0.0134 
Epoch 050 | loss 0.0150 
Epoch 060 | loss 0.0059 
Epoch 070 | loss 0.0083 
Epoch 080 | loss 0.0105 
Epoch 090 | loss 0.0066
```

预测结果：

```
User: hi, how are you?
AI: i'm doing great! thanks for asking. how about yourself?

User: what is the weather like today?
AI: please check a weather website or application for the current conditions.

User: can you recommend a good book?
AI: sure! to kill a mockingbird by harper lee is a classic and highly recommended novel.
```
