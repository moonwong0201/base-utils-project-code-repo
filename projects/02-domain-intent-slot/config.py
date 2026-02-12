import torch

class Args:
    train_path = './data/train_process.json'  # 训练集
    test_path = './data/test_process.json'    # 测试集
    domain_labels_path = './data/domains.txt'  # 领域标签
    seq_labels_path = './data/intents.txt'    # 意图标签
    token_labels_path = './data/slots.txt'    # 槽位标签
    bert_dir = 'google-bert/bert-base-chinese'
    save_dir = './checkpoints/'  # 训练后模型的保存目录
    load_dir = './checkpoints/model.pt'  # 预训练/已训练模型的加载路径
    do_train = True  # 是否执行训练任务
    do_eval = False  # 是否执行验证任务
    do_test = False  # 是否执行测试任务
    do_save = True   # 是否在训练结束后保存模型
    do_predict = True  # 是否执行单句预测任务
    load_model = True  # 是否加载已有模型
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")

    # 领域标签与ID的双向映射
    domlabel2id = {}
    id2domlabel = {}

    with open(domain_labels_path, 'r') as fp:
        domain_labels = fp.read().split('\n')
        for i, label in enumerate(domain_labels):
            domlabel2id[label] = i
            id2domlabel[i] = label

    # 意图标签与ID的双向映射（用于将文本标签转为数字ID，或反之）
    seqlabel2id = {}
    id2seqlabel = {}

    with open(seq_labels_path, 'r') as fp:
        seq_labels = fp.read().split('\n')
        for i, label in enumerate(seq_labels):
            seqlabel2id[label] = i
            id2seqlabel[i] = label

    # 槽位基础标签与ID的双向映射
    tokenlabel2id = {}
    id2tokenlabel = {}

    with open(token_labels_path, 'r') as fp:
        token_labels = fp.read().split('\n')
        for i, label in enumerate(token_labels):
            tokenlabel2id[label] = i
            id2tokenlabel[i] = label

    tmp = ['O']  # 先加入“非实体”标签O
    for label in token_labels:  # 遍历基础槽位标签，生成B-xxx和I-xxx格式的标签
        B_label = 'B-' + label
        I_label = 'I-' + label
        tmp.append(B_label)
        tmp.append(I_label)
    # 槽位BIO标签与ID的双向映射（用于槽位填充任务的标签转换）
    nerlabel2id = {}
    id2nerlabel = {}
    for i, label in enumerate(tmp):
        nerlabel2id[label] = i
        id2nerlabel[i] = label

    hidden_size = 768  # BERT模型的隐藏层维度
    domain_num_labels = len(domain_labels)  # 领域类别数量
    seq_num_labels = len(seq_labels)  # 意图类别数量
    token_num_labels = len(tmp)  # 槽位BIO标签数量
    max_len = 32  # 文本最大长度（超过会截断，不足会填充，需根据数据长度调整）
    batch_size = 64  # 每次训练的批量大小
    lr = 2e-5
    epoch = 10
    hidden_dropout_prob = 0.1
    hidden_dropout_prob_ner = 0.3

    API_KEY = API_KEY  # 替换成你的
    BASE_URL = BASE_URL

if __name__ == '__main__':
    args = Args()
    print(args.seq_labels)
    print(args.seqlabel2id)
    print(args.tokenlabel2id)
    print(args.nerlabel2id)
