from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import os
from seqeval.metrics.sequence_labeling import get_entities
# 从seqeval库导入实体提取工具，用于从槽位标注结果中提取实体

from config import Args  # 从自定义config模块导入Args类，用于存储训练参数
from model import BertForIntentClassificationAndSlotFilling
# 从自定义model模块导入模型类，这是一个同时处理意图识别和槽位填充的BERT模型

from dataset import BertDataset
from preprocess import Processor, get_features
# 从自定义preprocess模块导入数据处理器和特征提取函数，用于数据预处理


class Trainer:
    def __init__(self, model, config):
        self.model = model   # 保存模型实例
        self.config = config  # 保存配置参数
        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵损失
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)  # 初始化优化器：Adam，学习率从配置中获取
        self.epoch = self.config.epoch   # 训练轮数
        self.device = self.config.device   # 训练设备

    # 训练方法，接收训练数据加载器
    def train(self, train_loader):
        global_step = 0   # 记录全局训练步数（所有epoch的总步数）
        total_step = len(train_loader) * self.epoch  # 计算总训练步数：每轮步数×轮数
        self.model.train()  # 将模型切换到训练模式（启用 dropout 等训练特有的层）
        for epoch in range(self.epoch):
            total_loss = 0.0
            for step, train_batch in enumerate(train_loader):  # 循环每个batch的数据
                for key in train_batch.keys():
                    train_batch[key] = train_batch[key].to(self.device)
                input_ids = train_batch['input_ids']   # 文本转换后的数字ID（BERT的核心输入）
                attention_mask = train_batch['attention_mask']   # 注意力掩码（标记哪些位置是真实文本，哪些是填充的0）
                token_type_ids = train_batch['token_type_ids']   # 句子类型掩码（用于区分句子对，单句时全为0）
                domain_label_ids = train_batch['domain_label_ids']  # 领域标签
                seq_label_ids = train_batch['seq_label_ids']     # 意图标签
                token_label_ids = train_batch['token_label_ids']  # 槽位标签

                # 模型前向传播：输入数据，得到意图预测和槽位预测结果
                # seq_output 模型输出的槽位预测，形状为[batch_size, seq_len, token_num_labels]

                domain_output, seq_output, token_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

                # 处理槽位预测的损失：忽略填充部分（padding）的损失
                active_loss = attention_mask.view(-1) == 1  # 生成掩码：1表示真实文本位置，0表示填充位置
                active_logits = token_output.view(-1, token_output.shape[2])[active_loss]  # 只保留真实文本位置的预测结果
                active_labels = token_label_ids.view(-1)[active_loss]  # 只保留真实文本位置的真实标签

                # 计算损失
                domain_loss = self.criterion(domain_output, domain_label_ids)  # 领域识别损失（分类损失）
                seq_loss = self.criterion(seq_output, seq_label_ids)  # 意图识别损失（分类损失）
                token_loss = self.criterion(active_logits, active_labels)  # 槽位填充损失（分类损失，仅计算真实文本位置）
                loss = 0.2 * domain_loss + 0.2 * seq_loss + 1.0 * token_loss  # 总损失：意图损失+槽位损失

                # 反向传播更新参数
                self.model.zero_grad()   # 清空模型参数的梯度（避免累积）
                loss.backward()          # 反向传播计算梯度
                self.optimizer.step()    # 优化器根据梯度更新参数

                if step % 10 == 0:
                    print(f'[train] epoch:{epoch+1} {global_step}/{total_step} loss:{loss.item()}')

                total_loss += loss.item()

                global_step += 1  # 全局步数+1

            print(f'[train] epoch:{epoch + 1} loss:{total_loss / args.batch_size}')

        # 如果配置了保存模型，训练结束后保存模型参数
        if self.config.do_save:
            self.save(self.config.save_dir, 'model.pt')

    def test(self, test_loader):
        self.model.eval()  # 将模型切换到评估模式（关闭 dropout 等）
        domain_preds = []  # 领域预测结果
        domain_trues = []  # 领域真实标签
        seq_preds = []     # 意图预测结果
        seq_trues = []     # 意图真实标签
        token_preds = []   # 槽位预测结果
        token_trues = []   # 槽位真实标签
        with torch.no_grad():
            for step, test_batch in enumerate(test_loader):  # 循环每个batch的测试数据
                for key in test_batch.keys():
                    test_batch[key] = test_batch[key].to(self.device)
                # 提取输入
                input_ids = test_batch['input_ids']  # 文本转换后的数字ID（BERT的核心输入）
                attention_mask = test_batch['attention_mask']  # 注意力掩码（标记哪些位置是真实文本，哪些是填充的0）
                token_type_ids = test_batch['token_type_ids']  # 句子类型掩码（用于区分句子对，单句时全为0）
                domain_label_ids = test_batch['domain_label_ids']  # 领域标签
                seq_label_ids = test_batch['seq_label_ids']    # 意图标签
                token_label_ids = test_batch['token_label_ids']  # 槽位标签

                # 模型前向传播（不计算梯度）
                domain_output, seq_output, token_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

                # 处理领域识别结果
                domain_output = domain_output.detach().cpu().numpy()
                domain_output = np.argmax(domain_output, -1)   # 取概率最大的类别作为预测结果
                domain_label_ids = domain_label_ids.detach().cpu().numpy()  # 真实标签
                domain_label_ids = domain_label_ids.reshape(-1)  # 展平为一维数组
                domain_preds.extend(domain_output)    # 保存预测结果
                domain_trues.extend(domain_label_ids)   # 保存真实标签

                # 处理意图识别结果
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)   # 取概率最大的类别作为预测结果
                seq_label_ids = seq_label_ids.detach().cpu().numpy()  # 真实标签
                seq_label_ids = seq_label_ids.reshape(-1)  # 展平为一维数组
                seq_preds.extend(seq_output)    # 保存预测结果
                seq_trues.extend(seq_label_ids)   # 保存真实标签

                # 处理槽位填充结果
                token_output = token_output.detach().cpu().numpy()
                token_label_ids = token_label_ids.detach().cpu().numpy()
                token_output = np.argmax(token_output, -1)   # 取每个位置概率最大的类别
                # 计算每个句子的真实长度（去除填充部分）
                active_len = torch.sum(attention_mask, -1).view(-1)  # 每个句子中真实文本的长度（attention_mask为1的数量）

                # 遍历每个句子，处理预测和真实标签（去除BERT添加的首尾特殊符号）
                for length, t_output, t_label in zip(active_len, token_output, token_label_ids):
                    t_output = t_output[1:length-1]  # 截取有效部分：BERT会在句首加[CLS]、句尾加[SEP]，这里去除
                    t_label = t_label[1:length-1]    # 同上处理真实标签
                    t_ouput = [self.config.id2nerlabel[i] for i in t_output]  # 将数字ID转为槽位标签
                    t_label = [self.config.id2nerlabel[i] for i in t_label]   # 真实标签ID转文本
                    token_preds.append(t_ouput)  # 保存槽位预测结果
                    token_trues.append(t_label)  # 保存槽位真实标签

        # 计算领域识别的评估指标（准确率、精确率、召回率、F1）
        domain_acc, domain_precision, domain_recall, domain_f1 = self.get_metrices(domain_trues, domain_preds, 'cls')
        domain_report = self.get_report(domain_trues, domain_preds, 'cls')
        # 计算意图识别的评估指标（准确率、精确率、召回率、F1）
        intent_acc, intent_precision, intent_recall, intent_f1 = self.get_metrices(seq_trues, seq_preds, 'cls')
        intent_report = self.get_report(seq_trues, seq_preds, 'cls')
        ner_acc, ner_precision, ner_recall, ner_f1 = self.get_metrices(token_trues, token_preds, 'ner')
        ner_report = self.get_report(token_trues, token_preds, 'ner')

        print('领域识别：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            domain_acc, domain_precision, domain_recall, domain_f1
        ))
        print(domain_report)

        print('意图识别：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            intent_acc, intent_precision, intent_recall, intent_f1
        ))
        print(intent_report)

        print('槽位填充：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            ner_acc, ner_precision, ner_recall, ner_f1
        ))
        print(ner_report)

    # 计算评估指标的方法
    def get_metrices(self, trues, preds, mode):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        if mode == 'cls':  # 领域/意图识别（分类任务）
            acc = accuracy_score(trues, preds)  # 准确率：预测正确的样本数/总样本数
            precision = precision_score(trues, preds, average='micro')  # 精确率：预测为正且实际为正的比例（micro：多类别的全局计算）
            recall = recall_score(trues, preds, average='micro')  # 召回率：实际为正且被预测为正的比例
            f1 = f1_score(trues, preds, average='micro')  # F1值：精确率和召回率的调和平均
        elif mode == 'ner':  # 槽位填充（序列标注任务）
            # 1. Token级准确率（参考用）
            trues_flat = [label for seq in trues for label in seq]
            preds_flat = [label for seq in preds for label in seq]
            acc = accuracy_score(trues_flat, preds_flat) if trues_flat else 0.0

            # 2. 实体级精准率/召回率/F1（核心，用seqeval）
            from seqeval.metrics import precision_score as ner_precision
            from seqeval.metrics import recall_score as ner_recall
            from seqeval.metrics import f1_score as ner_f1
            precision = ner_precision(trues, preds, zero_division=0)
            recall = ner_recall(trues, preds, zero_division=0)
            f1 = ner_f1(trues, preds, zero_division=0)
        return acc, precision, recall, f1

    # 生成评估报告的方法
    def get_report(self, trues, preds, mode):
        if mode == 'cls':  # 意图识别报告（分类任务）
            from sklearn.metrics import classification_report
            report = classification_report(trues, preds)  # 生成包含每个类别的准确率、精确率等的报告
        elif mode == 'ner':   # 槽位填充报告（序列标注任务）
            from seqeval.metrics import classification_report
            report = classification_report(trues, preds)  # 生成序列标注的详细报告
        return report

    def save(self, save_path, save_name):
        # 将模型参数保存到指定路径（只保存参数，不保存整个模型，节省空间）
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))

    def predict(self, text):  # 对单条文本进行预测的方法
        self.model.eval()
        with torch.no_grad():
            tmp_text = [i for i in text]  # 将文本拆分为字符（按字符处理，也可按词，取决于预处理逻辑）
            # 使用分词器将文本转换为模型输入格式
            inputs = tokenizer.encode_plus(
                text=tmp_text,  # 输入文本（字符列表）
                max_length=self.config.max_len,  # 最大长度（超过截断，不足填充）
                padding='max_length',  # 填充到最大长度
                truncation='only_first',  # 截断方式：只截断第一个句子
                return_attention_mask=True,  # 返回注意力掩码
                return_token_type_ids=True,  # 返回句子类型掩码
                return_tensors='pt'  # 返回PyTorch张量
            )
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)

            # 提取输入（同训练/测试）
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']

            # 模型前向传播，得到预测结果
            domain_output, seq_output, token_output = self.model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            # 处理意图预测结果和槽位预测结果
            domain_output = domain_output.detach().cpu().numpy()
            seq_output = seq_output.detach().cpu().numpy()
            token_output = token_output.detach().cpu().numpy()
            domain_output = np.argmax(domain_output, -1)
            seq_output = np.argmax(seq_output, -1)
            token_output = np.argmax(token_output, -1)
            # print(seq_output, token_output)

            # 解析结果（提取有效部分）
            domain_output = domain_output[0]  # 取第一个样本（因为只预测了一条文本）
            seq_output = seq_output[0]
            # 截取槽位结果：去除BERT添加的首尾符号，长度与原文本一致
            token_output = token_output[0][1:len(text)-1]
            # 将槽位ID转为标签文本
            token_output = [self.config.id2nerlabel[i] for i in token_output]

            print('领域：', self.config.id2domlabel[domain_output])
            print('意图：', self.config.id2seqlabel[seq_output])
            # get_entities会解析标签序列，提取出实体类型和位置（如B-地点、I-地点组合为一个地点实体）
            print('槽位：', str([(i[0], text[i[1]: i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]))


if __name__ == '__main__':
    args = Args()
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    # 初始化模型：同时处理意图识别和槽位填充的BERT模型
    model = BertForIntentClassificationAndSlotFilling(args)
    model.to(args.device)

    trainer = Trainer(model, args)

    if os.path.exists(args.load_dir):
        model.load_state_dict(torch.load(args.load_dir, map_location=args.device))
        print(f"成功加载预训练模型：{args.load_dir}")
    else:
        print(f"未找到预训练模型：{args.load_dir}")
        print("开始从头训练模型...")
        # 加载并预处理训练数据
        raw_examples = Processor.get_examples(args.train_path, 'train')  # 从文件读取原始训练数据
        train_features = get_features(raw_examples, tokenizer, args)  # 将原始数据转换为模型需要的特征
        train_dataset = BertDataset(train_features)  # 包装为数据集
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 创建数据加载器（批量加载、打乱数据）

        # 初始化训练器并开始训练
        trainer.train(train_loader)

    # 加载并预处理测试数据
    raw_examples = Processor.get_examples(args.test_path, 'test')  # 读取原始测试数据
    test_features = get_features(raw_examples, tokenizer, args)  # 转换为模型特征
    test_dataset = BertDataset(test_features)  # 包装为数据集
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)  # 创建数据加载器（批量加载、打乱数据）
    # 用测试数据评估模型
    trainer.test(test_loader)
    # 对测试集中的部分文本进行单句预测（展示效果）
    with open('./data/test.json', 'r') as fp:
        pred_data = eval(fp.read())
        for i, p_data in enumerate(pred_data):  # 遍历测试数据
            text = p_data['text']
            print('=================================')
            print(text)  # 打印原始文本
            trainer.predict(text)  # 预测该文本的意图和槽位
            print('=================================')
            if i == 10:  # 只预测前11条（0到10）
                break
