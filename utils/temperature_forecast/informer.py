import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# 定义设置随机种子的函数，确保模型训练结果可复现（每次运行代码结果一致）
def setup_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch CPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch GPU的随机种子（多GPU也生效）
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)     # 设置Python原生random库的随机种子
    torch.backends.cudnn.deterministic = True  # 禁用cuDNN的随机化算法，保证结果可复现


setup_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")

dataset = pd.read_csv("daily-minimum-temperatures.csv")
dataset.columns = ["Date", "Temperatures"]
dataset['Temperatures'] = pd.to_numeric(dataset['Temperatures'], errors='coerce')
dataset['Temperatures'] = dataset['Temperatures'].fillna(method='ffill').fillna(method='bfill')
dataset['Date'] = pd.to_datetime(dataset['Date'])
DATA_START_DATE = dataset['Date'].iloc[0]


# 数据预处理
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, stride=1):
        # 初始化函数：接收原始时序数据、输入序列长度、预测序列长度、步长
        self.data = data           # 原始标准化后的时序数据（一维数组）
        self.seq_len = seq_len     # 输入模型的历史序列长度（用前seq_len个点预测未来）
        self.pred_len = pred_len   # 需要预测的未来序列长度（预测后pred_len个点）
        self.stride = stride       # 滑动窗口的步长，控制样本生成的密度

    def __len__(self):
        # 重写__len__方法：返回数据集的样本数量（滑动窗口生成样本）
        # 计算逻辑：总数据长度 - 输入序列长度 - 预测序列长度，除以步长后加1
        n = (len(self.data) - self.seq_len - self.pred_len) // self.stride + 1
        return max(0, n)  # 确保样本数量非负，避免无效值

    def __getitem__(self, idx):
        # 重写__getitem__方法：根据索引idx返回单个样本（输入序列+目标序列+时间特征）
        start_idx = idx * self.stride   # 计算当前样本的起始索引（按步长滑动）
        end_idx = start_idx + self.seq_len  # 输入序列的结束索引
        pred_end_idx = end_idx + self.pred_len  # 目标序列的结束索引

        # 提取输入序列x（历史数据）和目标序列y（待预测数据）
        seq_x = self.data[start_idx:end_idx]
        seq_y = self.data[end_idx:pred_end_idx]

        # 生成对应序列的日期范围，用于构建时间特征（年、月、日、星期）
        dates = pd.date_range(start=DATA_START_DATE + pd.Timedelta(days=start_idx),
                              periods=self.seq_len + self.pred_len, freq='D')

        # 初始化时间特征列表，用于存储每个日期的4个时间特征
        time_features = []
        for date in dates:
            # 对每个日期，提取年、月、日、星期几（weekday()返回0-6，对应周一到周日）
            time_features.append([
                (date.year - 1981) / 9,  # 年份归一化：1981-1990共9年，缩放到0-1
                date.month / 12,  # 月份归一化：缩放到0-1
                date.day / 31,  # 日期归一化：缩放到0-1
                date.weekday() / 6  # 星期归一化：缩放到0-1
            ])
        # 将时间特征列表转换为float32类型的numpy数组，符合PyTorch输入要求
        time_features = np.array(time_features, dtype=np.float32)

        # 拆分时间特征：输入序列对应的时间特征x_mark，目标序列对应的时间特征y_mark
        x_mark = time_features[:self.seq_len]
        y_mark = time_features[self.seq_len:]

        dec_input = np.zeros((self.pred_len, 1), dtype=np.float32)
        dec_input[:1, 0] = seq_x[-1]  # 第一个预测点用历史最后一个值

        # 返回字典格式的样本，包含输入序列、输入时间特征、目标序列、目标时间特征、真实标签
        return {
            'x_enc': torch.FloatTensor(seq_x).unsqueeze(-1),  # 输入序列x，增加最后一维（特征维度）
            'x_mark_enc': torch.FloatTensor(x_mark),  # 输入序列对应的时间特征
            'x_dec': torch.FloatTensor(dec_input),  # 解码器输入序列（此处用目标序列填充）
            'x_mark_dec': torch.FloatTensor(y_mark),  # 目标序列对应的时间特征
            'y': torch.FloatTensor(seq_y)  # 真实标签（待预测的气温序列）
        }


# 参数设置
seq_len = 200    # 输入序列长度：用前200天的气温数据作为历史输入
pred_len = 500   # 预测序列长度：预测未来500天的气温数据
batch_size = 32  # 批次大小：每次训练传入32个样本，平衡训练速度和内存占用

# 初始化StandardScaler标准化器，用于对气温数据进行标准化处理
scaler = StandardScaler()
# 对气温数据进行标准化：先reshape为二维数组（符合scaler要求），再拟合并转换，最后展平为一维数组
data_scaled = scaler.fit_transform(dataset['Temperatures'].values.reshape(-1, 1)).flatten()

# 划分训练集和测试集：确保测试集包含最后seq_len+pred_len个数据（用于最终预测验证）
train_size = len(data_scaled) - pred_len - seq_len
train_data = data_scaled[:train_size]   # 训练集：前train_size个标准化后的数据
test_data = data_scaled[train_size:]    # 测试集：剩余数据（包含输入序列和目标序列）

# 实例化自定义训练数据集，步长设为5（减少样本数量，提升训练速度）
train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len, stride=5)
# 实例化DataLoader数据迭代器：封装训练数据集，开启打乱顺序，关闭多线程加载
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# 简化版的Informer模型
class SimplifiedInformer(nn.Module):
    def __init__(self, enc_in=1, dec_in=1, c_out=1, seq_len=96, label_len=48, pred_len=20,
                 d_model=128, n_heads=4, e_layers=2, d_layers=1, d_ff=256,
                 dropout=0.1, device=device):
        super(SimplifiedInformer, self).__init__()
        self.pred_len = pred_len  # 预测序列长度
        self.d_model = d_model    # 模型嵌入维度（特征映射后的维度）

        # 编码器嵌入层：将输入序列（1维气温+4维时间特征：年、月、日、星期）映射到d_model维
        self.enc_embedding = nn.Linear(enc_in + 4, d_model)

        # 解码器嵌入层：将解码器输入序列（1维气温+4维时间特征）映射到d_model维
        self.dec_embedding = nn.Linear(dec_in + 4, d_model)

        # 位置编码层：为序列添加位置信息（Transformer本身无位置感知能力）
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)

        # 编码器：使用PyTorch的TransformerEncoder，由多个EncoderLayer组成
        self.encoder = nn.TransformerEncoder(
            # 单个编码器层配置：嵌入维度、注意力头数、前馈网络维度、 dropout率
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True  # 设置batch为第一维度（[batch, seq_len, d_model]）
            ),
            num_layers=e_layers   # 编码器层数
        )

        # 解码器：使用PyTorch的TransformerDecoder，由多个DecoderLayer组成
        self.decoder = nn.TransformerDecoder(
            # 单个解码器层配置：与编码器层参数对应
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True  # 设置batch为第一维度
            ),
            num_layers=d_layers   # 解码器层数
        )

        # 输出投影层：将解码器输出的d_model维特征映射为1维（气温预测值）
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 前向传播函数：定义模型数据流动路径
        # 编码器部分：拼接气温特征和时间特征，再嵌入+位置编码+编码
        enc_input = torch.cat([x_enc, x_mark_enc], dim=-1)  # 拼接：[batch, seq_len, 1+4]
        enc_embedded = self.enc_embedding(enc_input)   # 嵌入映射：[batch, seq_len, d_model]
        enc_embedded = self.pos_encoder(enc_embedded)  # 添加位置编码

        memory = self.encoder(enc_embedded)  # 编码器输出：作为解码器的记忆信息

        # 解码器部分：拼接气温特征和时间特征，再嵌入+位置编码+解码
        dec_input = torch.cat([x_dec, x_mark_dec], dim=-1)  # 拼接：[batch, pred_len, 1+4]
        dec_embedded = self.dec_embedding(dec_input)  # 嵌入映射：[batch, pred_len, d_model]
        dec_embedded = self.pos_encoder(dec_embedded)  # 添加位置编码

        output = self.decoder(dec_embedded, memory)  # 解码器输出：结合自身输入和编码器记忆
        output = self.projection(output)  # 投影到1维：[batch, pred_len, 1]

        return output[:, -self.pred_len:, :]  # 只返回预测长度的输出（确保输出长度与目标序列一致）


# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 初始化位置编码矩阵pe：[max_len, d_model]，存储所有位置的编码信息
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引：[max_len, 1]，表示每个位置的序号
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码的分母项（指数衰减）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # 偶数位置用正弦函数填充位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置用余弦函数填充位置编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度：[1, max_len, d_model]，方便与批量序列拼接
        pe = pe.unsqueeze(0)
        # 将pe注册为缓冲区（不参与模型参数更新，仅作为固定特征）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 前向传播：将输入序列x与对应长度的位置编码相加
        return x + self.pe[:, :x.size(1), :]


# 创建模型
model = SimplifiedInformer(
    enc_in=1,  # 编码器输入特征维度（气温为1维）
    dec_in=1,  # 解码器输入特征维度（气温为1维）
    c_out=1,   # 模型输出特征维度（预测气温为1维）
    seq_len=seq_len,     # 输入序列长度（200）
    label_len=pred_len,  # 标签序列长度（500）
    pred_len=pred_len,   # 预测序列长度（500）
    d_model=128,   # 嵌入维度（128）
    n_heads=4,     # 多头注意力头数（4）
    e_layers=2,    # 编码器层数（2）
    d_layers=1,    # 解码器层数（1）
    d_ff=256,      # 前馈网络隐藏层维度（256）
    dropout=0.1,   # Dropout正则化率（0.1，防止过拟合）
    device=device
).to(device)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义学习率调度器：ReduceLROnPlateau，当训练损失不再下降时，自动降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


# 定义单轮训练函数：完成一次训练集的迭代训练，返回平均损失
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()    # 将模型设为训练模式（启用Dropout等正则化）
    total_loss = 0   # 初始化总损失

    # 遍历DataLoader中的每个批次数据
    for batch in dataloader:
        x_enc = batch['x_enc'].to(device)
        x_mark_enc = batch['x_mark_enc'].to(device)
        x_dec = batch['x_dec'].to(device)
        x_mark_dec = batch['x_mark_dec'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 计算损失：预测输出与真实标签的MSE损失（标签增加最后一维，匹配输出形状）
        loss = criterion(outputs, y.unsqueeze(-1))

        # 反向传播
        loss.backward()
        # 梯度裁剪：限制梯度最大范数为1.0，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()  # 累加批次损失（item()提取张量的标量值）
    # 返回本轮训练的平均损失（总损失除以批次数量）
    return total_loss / len(dataloader)

num_epochs = 20
train_losses = []  # 初始化列表，用于存储每轮训练的损失值

for epoch in range(num_epochs):
    # 调用单轮训练函数，得到本轮训练损失
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    # 学习率调度器更新：传入本轮损失，判断是否需要降低学习率
    scheduler.step(train_loss)
    train_losses.append(train_loss)  # 记录本轮损失

    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

# 准备最终预测的数据：提取标准化后数据的最后seq_len+pred_len个点（用于预测未来500天）
last_data = data_scaled[-(seq_len + pred_len):]
last_seq = last_data[:seq_len]   # 输入序列：最后200天的标准化气温数据
last_target = last_data[seq_len:]   # 目标序列：最后500天的标准化气温数据

# 生成对应数据的日期范围，用于构建时间特征
last_dates = pd.date_range(
    start=DATA_START_DATE + pd.Timedelta(days=len(data_scaled) - seq_len - pred_len),
    periods=seq_len + pred_len,
    freq='D'
)

# 构建最后一批数据的时间特征（年、月、日、星期）
last_time_features = []
for date in last_dates:
    last_time_features.append([
        (date.year - 1981) / 9,  # 年份归一化：1981-1990共9年，缩放到0-1
        date.month / 12,  # 月份归一化：缩放到0-1
        date.day / 31,  # 日期归一化：缩放到0-1
        date.weekday() / 6  # 星期归一化：缩放到0-1
    ])
# 转换为float32类型的numpy数组
last_time_features = np.array(last_time_features, dtype=np.float32)

# 拆分时间特征为编码器和解码器对应的特征
x_mark_enc = last_time_features[:seq_len]
x_mark_dec = last_time_features[seq_len:]

# 【修复】预测阶段解码器输入：和训练阶段一致，用历史最后一个值初始化+0填充
dec_input_pred = np.zeros((pred_len, 1), dtype=np.float32)
dec_input_pred[:1, 0] = last_seq[-1]  # 用历史最后一个值初始化

# 将数据转换为PyTorch张量，并增加batch维度（变为[1, seq_len, 1]），移动到指定设备
x_enc_tensor = torch.FloatTensor(last_seq).unsqueeze(0).unsqueeze(-1).to(device)
x_mark_enc_tensor = torch.FloatTensor(x_mark_enc).unsqueeze(0).to(device)
x_dec_tensor = torch.FloatTensor(dec_input_pred).unsqueeze(0).to(device)
x_mark_dec_tensor = torch.FloatTensor(x_mark_dec).unsqueeze(0).to(device)

# 进行预测
model.eval()
with torch.no_grad():
    pred_scaled = model(x_enc_tensor, x_mark_enc_tensor, x_dec_tensor, x_mark_dec_tensor)

# 反标准化：将预测结果和真实结果恢复为原始气温尺度
pred_scaled_np = pred_scaled.cpu().numpy().reshape(-1, 1)  # 转为numpy数组并调整形状
pred_original = scaler.inverse_transform(pred_scaled_np).flatten()  # 反标准化后展平
actual_original = scaler.inverse_transform(last_target.reshape(-1, 1)).flatten()   # 真实值反标准化后展平

# 创建结果汇总DataFrame，存储日期、真实气温、预测气温
results = pd.DataFrame({
    'Date': last_dates[seq_len:],  # 预测对应的日期
    'Actual': actual_original,     # 原始真实气温
    'Predicted': pred_original     # 模型预测气温
})

# 计算最后500天预测的评估指标，量化模型性能
last_mse = mean_squared_error(results['Actual'], results['Predicted'])
last_rmse = np.sqrt(last_mse)  # 均方根误差（RMSE），量纲与原始数据一致
last_mae = mean_absolute_error(results['Actual'], results['Predicted'])
# 平均绝对百分比误差（MAPE），反映相对误差大小
last_mape = np.mean(np.abs((results['Actual'] - results['Predicted']) / results['Actual'])) * 100

print("\n最后500天的预测性能指标:")
print(f"均方误差 (MSE): {last_mse:.4f}")
print(f"均方根误差 (RMSE): {last_rmse:.4f}")
print(f"平均绝对误差 (MAE): {last_mae:.4f}")
print(f"平均绝对百分比误差 (MAPE): {last_mape:.2f}%")

# 可视化
plt.figure(figsize=(15, 8))

# 完整序列
# 第一个子图：2行1列的第1个，展示完整气温时间序列
plt.subplot(2, 1, 1)
plt.plot(dataset['Date'], dataset['Temperatures'], label='Full Series', alpha=0.7, linewidth=1)
split_idx = len(dataset) - pred_len
plt.axvline(x=dataset['Date'].iloc[split_idx], color='red', linestyle='--',
            label='Train/Test Split', alpha=0.7)
plt.title(f'Daily Minimum Temperature Series (Total: {len(dataset)} days)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True, alpha=0.3)

# 第二个子图：2行1列的第2个，展示最后500天的预测结果与真实值对比
plt.subplot(2, 1, 2)
plt.plot(results['Date'], results['Actual'], 'b-', label='Actual', alpha=0.7, linewidth=1)
plt.plot(results['Date'], results['Predicted'], 'r-', label='Predicted', alpha=0.7, linewidth=1)
plt.title(f'Informer Model Predictions (Last {pred_len} Days)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()  # 自动调整子图间距，避免标签重叠
plt.show()
