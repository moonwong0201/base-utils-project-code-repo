import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
# 自动搜索最优的ARIMA模型参数，无需手动试错
from pmdarima import auto_arima
# 从sklearn.metrics（机器学习评估指标模块）中导入均方误差、平均绝对误差，用于评估预测模型的性能
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

dataset = pd.read_csv("daily-minimum-temperatures.csv")
dataset.columns = ["Date", "Temperatures"]

# 将"Temperatures"列的数据转换为数值类型（int/float），errors='coerce'表示转换失败时不报错，而是将无效值设为NaN（缺失值）
dataset['Temperatures'] = pd.to_numeric(dataset['Temperatures'], errors='coerce')
# 对"Temperatures"列的缺失值进行填充，method='ffill'表示前向填充（用前一个非缺失值填充当前缺失值），保证数据完整性
dataset['Temperatures'] = dataset['Temperatures'].fillna(method='ffill')
# 将"Date"列的数据转换为datetime日期类型，方便后续按日期索引、筛选时间序列数据
dataset['Date'] = pd.to_datetime(dataset['Date'])
# 将"Date"列设置为数据集的索引（索引用于快速查找数据，时间序列数据通常以日期为索引），inplace=True表示直接在原数据集上修改，不返回新数据集
dataset.set_index('Date', inplace=True)

train_size = len(dataset) - 500
train_data = dataset['Temperatures'].iloc[:train_size]
test_data = dataset['Temperatures'].iloc[train_size:]

# 调用auto_arima函数自动训练并筛选最优ARIMA模型，赋值给auto_model
auto_model = auto_arima(
    train_data,
    start_p=0,       # AR（自回归项）的最小阶数，从0开始搜索
    start_q=0,       # MA（移动平均项）的最小阶数，从0开始搜索
    max_p=5,         # AR项的最大阶数，搜索上限为5
    max_q=5,         # MA项的最大阶数，搜索上限为5
    max_d=2,         # d（差分阶数，用于使数据平稳）的最大阶数，搜索上限为2
    start_P=0,       # 季节性AR项的最小阶数，从0开始搜索（当前不启用季节性，此参数暂不生效）
    start_Q=0,       # 季节性MA项的最小阶数，从0开始搜索（当前不启用季节性，此参数暂不生效）
    max_P=2,         # 季节性AR项的最大阶数，搜索上限为2（当前不启用季节性，此参数暂不生效）
    max_Q=2,         # 季节性MA项的最大阶数，搜索上限为2（当前不启用季节性，此参数暂不生效）
    m=7,             # 季节性周期长度，设为1表示无明显季节性周期（当前不启用季节性，此参数暂不生效）
    seasonal=True,  # 是否考虑数据的季节性特征，False表示不考虑季节性，使用普通ARIMA模型（而非SARIMA）
    test='adf',      # 用于检验数据平稳性的方法，'adf'表示ADF检验（单位根检验），用于自动确定最优d值
    trace=True,      # 是否显示模型搜索过程中的详细信息，True表示打印每一步的候选模型、评估指标等
    error_action='ignore',   # 模型训练过程中遇到错误时的处理方式，'ignore'表示忽略错误并继续执行
    suppress_warnings=True,  # 是否抑制模型训练过程中产生的警告信息，True表示不打印警告
    stepwise=True,   # 是否使用逐步搜索策略寻找最优模型，True表示使用（速度更快，适合大部分场景）
    n_fits=50        # 最大拟合模型的次数，最多尝试50个候选模型，避免过度搜索
)

print("\n模型摘要:")
# 打印最优ARIMA模型的详细摘要信息，包括模型参数、系数、显著性检验（p值）、AIC/BIC等评估指标
print(auto_model.summary())

# 获取最佳参数
best_order = auto_model.order
print(f"\n最佳参数 (p,d,q): {best_order}")
# 模型摘要:
#                                SARIMAX Results
# ==============================================================================
# Dep. Variable:                      y   No. Observations:                 3150
# Model:               SARIMAX(3, 0, 1)   Log Likelihood               -7261.195
# Date:                Sun, 08 Feb 2026   AIC                          14534.390
# Time:                        16:57:14   BIC                          14570.721
# Sample:                             0   HQIC                         14547.426
#                                - 3150
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# intercept      0.0521      0.019      2.739      0.006       0.015       0.089
# ar.L1          1.4907      0.020     75.316      0.000       1.452       1.530
# ar.L2         -0.6212      0.029    -21.732      0.000      -0.677      -0.565
# ar.L3          0.1259      0.019      6.628      0.000       0.089       0.163
# ma.L1         -0.8949      0.013    -71.478      0.000      -0.919      -0.870
# sigma2         5.8817      0.139     42.191      0.000       5.608       6.155
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                14.47
# Prob(Q):                              0.96   Prob(JB):                         0.00
# Heteroskedasticity (H):               0.89   Skew:                             0.08
# Prob(H) (two-sided):                  0.07   Kurtosis:                         3.29
# ===================================================================================
# Warnings:
# [1] Covariance matrix calculated using the outer product of gradients (complex-step).

# 定义预测步数：等于测试集的长度（500步），即对测试集对应的时间区间进行预测
forecast_steps = len(test_data)
# 调用最优模型的predict方法进行预测，返回预测值和置信区间
forecast, conf_int = auto_model.predict(
    n_periods=forecast_steps,   # 预测的步数，即预测未来多少个时间点（此处为500步，对应测试集长度）
    return_conf_int=True,       # 是否返回预测结果的置信区间，True表示返回
    alpha=0.05                  # 置信水平对应的alpha值，0.05表示返回95%置信区间（置信区间反映预测的不确定性）
)

# 创建一个新的DataFrame（表格），用于存储测试集的实际值、预测值和置信区间
results = pd.DataFrame({
    'Date': test_data.index,      # 日期列：使用测试集的日期索引（保证日期对应）
    'Actual': test_data.values,   # 实际值列：测试集的真实温度值
    'Predicted': forecast,        # 预测值列：模型输出的温度预测值
    'Lower_CI': conf_int[:, 0],   # 置信区间下限列：取conf_int的第0列（所有行的第1个值）
    'Upper_CI': conf_int[:, 1]    # 置信区间上限列：取conf_int的第1列（所有行的第2个值）
})
# 将"Date"列设置为results数据集的索引，保持与原数据集一致的时间索引格式，方便后续可视化
results.set_index('Date', inplace=True)

# 计算模型预测性能指标：均方误差（MSE），反映预测值与实际值的平均平方偏差
mse = mean_squared_error(results['Actual'], results['Predicted'])
# 计算均方根误差（RMSE），是MSE的平方根，量纲与原始数据一致，更易理解误差大小
rmse = np.sqrt(mse)
# 计算平均绝对误差（MAE），反映预测值与实际值的平均绝对偏差，对异常值的鲁棒性强于MSE
mae = mean_absolute_error(results['Actual'], results['Predicted'])
# 计算平均绝对百分比误差（MAPE），反映预测误差的相对百分比，便于跨数据集比较，最后乘以100转为百分比格式
mape = np.mean(np.abs((results['Actual'] - results['Predicted']) / results['Actual'])) * 100

print("\n" + "="*60)
print("预测性能指标:")
print("="*60)
print(f"最佳模型: ARIMA{best_order}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

# 可视化结果
plt.figure(figsize=(15, 8))

# 绘制完整序列
plt.subplot(2, 1, 1)
# 绘制完整数据集的温度序列：蓝色线条（默认），标签为"Full Series"，透明度0.7，线宽1
plt.plot(dataset['Temperatures'], label='Full Series', alpha=0.7, linewidth=1)
# 绘制一条垂直虚线，位置在训练集最后一个日期，颜色红色，线型为虚线，标签为"Train/Test Split"，用于区分训练集和测试集
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
plt.title(f'ARIMA Model Predictions (Last 20 Days)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加误差线
for i, (date, row) in enumerate(results.iterrows()):
    plt.plot([date, date], [row['Actual'], row['Predicted']], 'gray', alpha=0.5, linewidth=0.5)

plt.tight_layout()
plt.show()
