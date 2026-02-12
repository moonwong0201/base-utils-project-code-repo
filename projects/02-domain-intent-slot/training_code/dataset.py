"""
连接预处理特征和模型训练的关键桥梁，核心作用是把前序生成的 InputFeature 特征列表封装成
PyTorch 可直接使用的 Dataset 数据集，再通过 DataLoader 实现批量加载、打乱等功能
"""
from torch.utils.data import DataLoader, Dataset
from config import Args

args = Args()

class BertDataset(Dataset):
    def __init__(self, features):  # 初始化方法，接收前序生成的InputFeature列表
        self.features = features   # 存储所有InputFeature实例（每个实例对应一条样本的特征）
        self.nums = len(self.features)  # 计算样本总数（数据集大小）

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        # item：样本索引（从0到self.nums-1）
        # 从InputFeature中提取各特征，并转换为long类型（PyTorch模型通常要求输入为长整型）
        data = {
            # 文本数字ID：对应BERT的核心输入，转换为long类型（避免默认int32可能的精度问题）
            'input_ids': self.features[item].input_ids.long(),
            # 注意力掩码：标记有效文本位置，同样转为long类型
            'attention_mask': self.features[item].attention_mask.long(),
            # 句子类型掩码：区分单句/句子对，转为long类型
            'token_type_ids': self.features[item].token_type_ids.long(),
            # 领域标签ID
            'domain_label_ids': self.features[item].domain_label_ids.long(),
            # 意图标签ID：句子级分类标签，转为long类型（CrossEntropyLoss要求标签为long）
            'seq_label_ids': self.features[item].seq_label_ids.long(),
            # 槽位标签ID：token级分类标签，转为long类型
            'token_label_ids': self.features[item].token_label_ids.long(),
        }
        return data  # 返回单条样本的特征字典（key为特征名，value为对应张量）


if __name__ == '__main__':
    from config import Args
    from preprocess import Processor, get_features
    from transformers import BertTokenizer
    args = Args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    raw_examples = Processor.get_examples(args.train_path, 'train')
    train_features = get_features(raw_examples, tokenizer, args)  # InputExample→InputFeature（预处理）
    train_dataset = BertDataset(train_features)  # InputFeature→BertDataset（封装为PyTorch数据集）
    # 基于数据集创建DataLoader：实现批量加载、打乱
    train_loader = DataLoader(
        train_dataset,  # 输入数据集
        batch_size=args.batch_size,  # 每批样本数
        shuffle=True
    )

    for step, train_batch in enumerate(train_loader):
        print(train_batch)
        break

    # 生成测试集数据集与DataLoader
    raw_examples = Processor.get_examples(args.test_path, 'test')
    test_features = get_features(raw_examples, tokenizer, args)
    test_dataset = BertDataset(test_features)
    # 测试集DataLoader：通常不打乱（方便后续对应原始样本分析）
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
