import os
import torch
import pytorch_lightning as pl
from functools import partial

# 强制把所有torch.load的weights_only设为False
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

original_torch_load = torch.load
torch.load = patched_torch_load
# 设置环境变量，强制torch.load默认weights_only=False
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
# 从neuralprophet库中导入NeuralProphet模型（深度学习时间序列预测模型）和set_log_level（控制日志级别）
from neuralprophet import NeuralProphet, set_log_level
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

dataset = pd.read_csv("daily-minimum-temperatures.csv")
dataset.columns = ["ds", "y"]  # NeuralProphet需要的列名
dataset['y'] = pd.to_numeric(dataset['y'], errors='coerce')
dataset['y'] = dataset['y'].fillna(method='ffill').fillna(method='bfill')
dataset['ds'] = pd.to_datetime(dataset['ds'])

train_size = len(dataset) - 500
train_data = dataset.iloc[:train_size].copy()
test_data = dataset.iloc[train_size:].copy()
print(test_data)

# 初始化NeuralProphet模型，配置模型核心参数
model = NeuralProphet(
    # 基础配置
    epochs=100,     # 模型训练的轮数：整个训练集将被迭代100次，用于优化模型参数
    n_forecasts=1,  # 单次预测的步长，设为1表示每次预测1个时间点（单日气温）
    # changepoints_range=0.8,  # 允许趋势变化点的范围（占训练集的80%），当前未启用
    # n_changepoints=20,       # 趋势变化点的数量，当前未启用，用于捕捉非线性趋势
    yearly_seasonality=True,   # 启用年度季节性：捕捉气温的年周期规律（如每年夏季热、冬季冷）
    weekly_seasonality=True,   # 启用周季节性：捕捉气温的周周期规律（针对日度数据）
    daily_seasonality=False,   # 关闭日季节性：每日最低气温无明显日内波动规律，无需启用
)

# 训练NeuralProphet模型：传入训练集数据，指定时间频率为'D'（按天），训练进度显示为'bar'（进度条）
# 返回的metrics是训练过程中的指标（如损失值），可用于查看模型训练效果
metrics = model.fit(
    train_data,
    freq='D',
    progress='bar',
    learning_rate=None
)

# 构建未来预测所需的数据集：基于训练集数据，生成len(test_data)（500）个未来时间点，
# 同时包含n_historic_predictions=len(test_data)个历史预测点，用于对齐测试集时间范围
future = model.make_future_dataframe(train_data, periods=len(test_data), n_historic_predictions=len(test_data))
# 使用训练好的模型对future数据集进行预测，返回包含预测值、时间列的DataFrame
forecast = model.predict(future)
# 提取测试集对应的预测结果：取forecast的最后500条数据（对应测试集时间范围），深拷贝避免修改原预测结果
forecast_test = forecast.iloc[-len(test_data):].copy()
# 创建结果汇总DataFrame，整合测试集的索引、实际气温值和模型预测值
results = pd.DataFrame({
    'Date': list(test_data.index),   # 日期列：使用测试集的行索引（整数索引）
    'Actual': list(test_data["y"].values),   # 实际值列：测试集的真实气温数据
    'Predicted': list(forecast_test["yhat1"].values),  # 预测值列：NeuralProphet预测的气温值（yhat1对应n_forecasts=1的预测结果）
})
# 将Date列设置为results的索引，方便后续可视化时按索引对齐数据
results.set_index('Date', inplace=True)


# 提取预测结果
predictions = forecast[['ds', 'y', 'yhat1']].tail(len(test_data)).copy()
# 重命名列名，方便后续理解：ds→Date，y→Actual，yhat1→Predicted
predictions.columns = ['Date', 'Actual', 'Predicted']
# 将Date列设置为predictions的索引
predictions.set_index('Date', inplace=True)

# 用测试集的真实气温值覆盖predictions中的Actual列（修正原有历史y值，确保与测试集实际值一致）
predictions['Actual'] = test_data['y'].values

# 计算评估指标
mse = mean_squared_error(predictions['Actual'], predictions['Predicted'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(predictions['Actual'], predictions['Predicted'])
mape = np.mean(np.abs((predictions['Actual'] - predictions['Predicted']) / predictions['Actual'])) * 100

print("\n" + "="*60)
print("预测性能指标")
print("="*60)
print(f"均方误差 (MSE): {mse:.4f}")  # 6.8001
print(f"均方根误差 (RMSE): {rmse:.4f}")  # 2.6077
print(f"平均绝对误差 (MAE): {mae:.4f}")  # 2.0717
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")  # 23.12%

plt.figure(figsize=(15, 8))

# 绘制完整序列
plt.subplot(2, 1, 1)
plt.plot(dataset['y'], label='Full Series', alpha=0.7, linewidth=1)
plt.axvline(x=train_data.index[-1], color='red', linestyle='--', label='Train/Test Split')
plt.title('Daily Minimum Temperature Series')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True, alpha=0.3)

# 绘制预测结果
plt.subplot(2, 1, 2)
plt.plot(results.index, results['Actual'], 'b-', label='Actual Values', marker='o', markersize=5, linewidth=1.5)
plt.plot(results.index, results['Predicted'], 'r--', label='Predicted Values', marker='s', markersize=5, linewidth=1.5)
plt.title(f'Prophet Model Predictions (Last 20 Days)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加误差线
for i, (date, row) in enumerate(results.iterrows()):
    plt.plot([date, date], [row['Actual'], row['Predicted']], 'gray', alpha=0.5, linewidth=0.5)

plt.tight_layout()
plt.show()
