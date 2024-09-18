import time
import random


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore") #过滤掉警告的意思
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（解决中文字体显示问题）
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号‘-’显示为方块的问题


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical





# 数据处理
# 读取数据
data = pd.read_csv('data.csv')
print('数据集预览如下: ')
time.sleep(2)
data.info()
print('\n'*5)

# 分离特征和标签
X = data.drop(columns=['Label', 'ID'])
y = data['Label']

# 标准化特征(可先进行数据可视化分析后再取消注释运行)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('数据清洗完毕后的数据集预览如下: ')
time.sleep(2)
X.info()
print('\n'*5)
# X_scaled = X

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 数据备份
init_data = []
init_data.append(X_train)
init_data.append(y_train)




# 训练集可视化数据分析和数据探索
train_data = pd.DataFrame(X_train)
train_data['label'] = y_train
print('训练集预览如下: ')
time.sleep(2)
train_data.info()
print('\n'*5)
print('以下为训练集的特征名字: ')
print(train_data.columns)
time.sleep(2)


# 绘制训练集的直方图(全体, 清晰度可能较低)
plt.figure(figsize=(20, 15))  # 设置图像大小
for i, column in enumerate(train_data.columns, 1):
    plt.subplot(4, 8, i)
    plt.hist(train_data[column], bins=20, color='blue', alpha=0.7, density=True)
    plt.title(f'character {i}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



# 为了清晰度更高, 需要输入单个属性来绘制直方图
flag = '1'
print('为了获得更高的清晰度，需要输入任意一个特征来绘制它们之间的直方图，注意特征需要带引号')
while eval(flag):
    plt.figure(figsize=(8, 6))
    x = eval(input('请输入单个特征'))
    print('\n')
    time.sleep(2)

    plt.hist(train_data[x], bins=20, color='blue', alpha=0.7)
    plt.title(f'特征 {x} 的直方图')
    plt.xlabel(f'Value')
    plt.ylabel(f'Frequency')
    plt.show()

    flag = input('若想继续观察数据, 输入1, 否则输入0')
    print('\n'*3)



# 绘制训练集的热力图
# 计算相关性矩阵
corr = train_data.corr()

plt.figure(figsize=(20, 15))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('相关性矩阵热力图')
plt.show()



# 为了清晰度更高, 需要输入两个属性来绘制二元关系散点图
flag = '1'
print('为了获得更高的清晰度，需要输入任意两个特征来绘制它们之间的二元关系散点图')
while eval(flag):
    plt.figure(figsize=(8, 6))
    x = eval(input('请输入第一个特征'))
    y = eval(input('请输入第二个特征'))
    print('\n')
    time.sleep(2)

    sns.scatterplot(x=train_data[x], y=train_data[y])
    plt.title(f'特征 {x} 和特征 {y} 的二元关系散点图')
    plt.xlabel(f'column {x}')
    plt.ylabel(f'column {y}')
    plt.show()

    flag = input('若想继续观察数据, 输入1, 否则输入0')
    print('\n'*3)








# 构建不同的神经网络模型
print('以下为普通FNN(前馈神经网络)')
time.sleep(2)

# 普通FNN
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型(浅层神经网络过拟合风险不大，不用验证集减少计算量)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'普通前馈神经网络模型准确率: {accuracy:.4f}')





# 深层前馈神经网络(10隐层)
print('\n'*5)
print('以下为深层前馈神经网络(10隐层)')
time.sleep(2)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'深度前馈神经网络模型准确率: {accuracy:.4f}')



# RNN循环神经网络
# 简单RNN
print('\n'*5)
print('以下为简单RNN')
time.sleep(2)

# RNN对数据有要求, 需转换数据格式
# 将特征转换为三维数组，符合RNN输入要求 (样本数量, 时间步长, 特征数量)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # 假设每个样本只有一个时间步

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# 创建RNN模型
model = Sequential()
model.add(SimpleRNN(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型(层数较少,过拟合风险不大，不使用验证集减少计算量)
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'简单RNN模型准确率: {accuracy:.4f}')




# 深度RNN
print('\n'*5)
print('以下为深度RNN')
time.sleep(2)

model = Sequential()
# 添加10个SimpleRNN层作为隐层
for _ in range(8):
    model.add(SimpleRNN(50, activation='relu', return_sequences=True))  # return_sequences=True 是为了让输出序列传递到下一个RNN层

# 添加最后一个SimpleRNN层，不需要返回序列
model.add(SimpleRNN(50, activation='relu'))
# 添加输出层
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'深度RNN模型准确率: {accuracy:.4f}')





# 深度LSTM模型
print('\n'*5)
print('以下为深度LSTM模型')

# 数据预处理(lstm的数据输入有格式限制，重新处理)
# X = data.drop(columns=['Label', 'ID'])
# y = data['Label']
# scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据调整为LSTM输入格式 (samples, timesteps, features)
# 这里我们假设每个样本是一个时间步长
X_scaled = np.expand_dims(X_scaled, axis=1)

# 将标签转换为分类格式
y_categorical = to_categorical(y)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_categorical, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



# 单向LSTM
print('单向LSTM')
time.sleep(2)

model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
for _ in range(8):  # 添加8层LSTM，共9层
    model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))  # 第10层LSTM
model.add(Dense(64, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'深度单向LSTM模型准确率: {accuracy:.4f}')



# 双向LSTM
print('\n'*5)
print('双向LSTM')
time.sleep(2)

# 构建双向LSTM模型
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
for _ in range(8):  # 添加8层双向LSTM，共9层
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64, return_sequences=False)))  # 第10层双向LSTM
model.add(Dense(64, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'深度双向LSTM模型准确率: {accuracy:.4f}')


