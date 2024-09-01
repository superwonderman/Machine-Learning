import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import warnings
warnings.filterwarnings("ignore") #过滤掉警告的意思
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（解决中文字体显示问题）
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号‘-’显示为方块的问题

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical






# 数据清洗和数据集处理

# 加载数据集
file_path = 'data.csv'
data = pd.read_csv(file_path)
# 加上属性名
data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
print(data.columns)

# 性别属性值编码
data = data.replace({'Sex': {'M': 0, 'F': 1, 'I': 2}})
# 年龄分段，0表示幼年，1表示成年，2表示老年
for i in data.index:
    if data.iloc[i, 8] <= 8:
        data.iloc[i, 8] = 0
    elif 9 <= data.iloc[i, 8] and data.iloc[i, 8] <= 12:
        data.iloc[i, 8] = 1
    else:
        data.iloc[i, 8] = 2


print('数据集预览如下: ')
time.sleep(2)
data.info()
time.sleep(2)
# 分割数据集
labels = data['Rings']
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3,
                                                                    random_state=123)

test_data.drop('Rings', axis=1, inplace=True)
print('\n'*5)
print('测试集预览如下: ')
test_data.info()
time.sleep(2)






# # 数据可视化分析和数据探索
#
# # 绘制训练集的直方图
# plt.figure(figsize=(15, 10))
#
# for i, column in enumerate(train_data.columns, 1):
#     plt.subplot(3, 3, i)
#     plt.hist(train_data[column], bins=10, edgecolor='k', alpha=0.7)
#     plt.title(column)
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#
# plt.tight_layout()
# plt.show()
#
#
# #绘制训练集的二元关系散点图
# plt.figure(figsize=(15, 15))
#
# for i, column1 in enumerate(train_data.columns):
#     for j, column2 in enumerate(train_data.columns):
#         plt.subplot(len(train_data.columns), len(train_data.columns), i * len(train_data.columns) + j + 1)
#         if i != j:
#             plt.scatter(data[column1], data[column2], s=10)
#             plt.xlabel(column1)
#             plt.ylabel(column2)
#         else:
#             plt.hist(data[column1], bins=10)
#             plt.xlabel(column1)
#             plt.ylabel("Frequency")
#         plt.tight_layout()
#
# plt.show()
#
#
# # 绘制训练集的热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('相关性矩阵热力图')
# plt.show()





# 训练集处理
train_data.drop('Rings', axis=1, inplace=True)
print('\n'*5)
print('训练集预览如下: ')
time.sleep(2)
train_data.info()
time.sleep(2)





# 数据集备份
init_train_data = train_data
init_train_labels = train_labels
init_test_data = test_data
init_test_labels = test_labels


train_data = init_train_data
train_labels = init_train_labels
test_data = init_test_data
test_labels = init_test_labels





# 选择不同模型进行建模分析

# 线性回归
print('\n'*5)
print('以下为线性回归模型: ')
time.sleep(2)

lr = LinearRegression()
lr.fit(train_data, train_labels)
lr_pred = lr.predict(test_data)


# 线性回归预测值数据转换
lr_pred[lr_pred <= 0.5] = 0
lr_pred[(lr_pred > 0.5) & (lr_pred <= 1.5)] = 1
lr_pred[lr_pred > 1.5] = 2

# 输出模型的斜率和截距
print('以下为线性回归模型参数')
print(f"线性回归的权重如下: {lr.coef_}") # 权重
print(f"线性回归的偏置如下: {lr.intercept_}") # 偏置

# 性能度量
lr_mse = np.mean((test_labels.astype(float) - lr_pred) ** 2)
print("线性回归的均方误差（MSE）：", lr_mse)

lr_accuracy = accuracy_score(test_labels, lr_pred)
print(f'线性回归准确率为：{lr_accuracy}')

plt.figure(figsize=(10, 6))
plt.bar(train_data.columns, lr.coef_)
plt.xlabel('属性')
plt.ylabel('权重')
plt.title('线性回归的权重')
plt.show()





# KNN最近邻分类
print('\n'*5)
print('以下为KNN最近邻分类: ')
time.sleep(2)

# 特征标准化
scaler = StandardScaler()
k_train_data = scaler.fit_transform(train_data) # 经过标准化处理数据集不再为DataFrame格式，需重新转换格式

knn_neighbors = [2, 3, 4, 5, 6, 7, 8, 9, 10]
k_train_data = pd.DataFrame(k_train_data)
k_train_data.info()
knn_best = []
knn_best_neighbors = 0
knn_best_accuracy = 0

knn_train_data, knn_test_data, knn_train_labels, knn_test_labels = train_test_split(k_train_data, train_labels,
                                                                                    test_size=0.3,
                                                                                    random_state=123)

# 交叉验证
for j in knn_neighbors:
    knn_ = KNeighborsClassifier(n_neighbors=j)
    knn_.fit(knn_train_data, knn_train_labels)
    knn_pred_ = knn_.predict(knn_test_data)
    knn_best_accuracy_ = accuracy_score(knn_pred_, knn_test_labels)
    if knn_best_accuracy_ > knn_best_accuracy:
        knn_best_accuracy = knn_best_accuracy_
        knn_best = knn_

# 使用模型预测测试集
knn_pred = knn_best.predict(test_data)

# 计算准确率
knn_accuracy = accuracy_score(knn_pred, test_labels)
print(f"KNN最近邻法分类模型的准确率为: {knn_accuracy}")






# softmax回归
train_data = init_train_data
train_labels = init_train_labels
test_data = init_test_data
test_labels = init_test_labels
print('\n'*5)
print('以下为softmax回归: ')
time.sleep(2)

# 使用softmax回归进行分类
softmax_model = LogisticRegression(multi_class='multinomial', solver='saga', random_state=0)
softmax_model.fit(train_data, train_labels)

# softmax_pred = softmax_model.predict(test_data)
# print(softmax_pred)

softmax_accuracy = softmax_model.score(test_data, test_labels)
print(f'softmax回归的准确率为: {softmax_accuracy}')

softmax_weights = softmax_model.coef_
num_classes = softmax_weights.shape[0]
num_features = train_data.shape[1]

print(f'softmax回归模型的权重为: {softmax_weights}')

# softmax回归权重可视化
plt.figure(figsize=(14, 7))

# 为每个类别绘制权重
for class_index in range(num_classes):
    plt.subplot(1, num_classes, class_index + 1)
    plt.bar(range(num_features), softmax_weights[class_index])
    plt.title(f'类别 {class_index} 的权重')
    plt.xlabel('属性')
    plt.ylabel('权重')

# 调整布局并显示图形
plt.tight_layout()
plt.show()





# 神经网络模型
train_data = init_train_data
train_labels = init_train_labels
test_data = init_test_data
test_labels = init_test_labels
print('\n'*5)
print('以下为神经网络: ')
time.sleep(2)
# 定义神经网络模型
network_model = Sequential([
    InputLayer(input_shape=(train_data.shape[1],)),  # 输入层，接受处理后的特征矩阵
    Dense(8, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    # Dense(8, activation='sigmoid'),
    # Dense(8, activation='sigmoid'),
    Dense(3, activation='softmax')  # 输出层，只有一个神经元，因为没有类别之分，而是直接预测数值
])

train_labels = to_categorical(train_labels)
sgd = SGD(learning_rate=0.05)
# 编译模型
network_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
network_model.fit(train_data, train_labels, epochs=50, batch_size=32)

# # 评估模型性能
# test_labels = to_categorical(test_labels)
# model_pred = network_model.predict(test_data)

# for i in model_pred:
#     print(i)

# print(type(model_pred))
# print(type(test_labels))
# model_mse = np.mean((test_labels.values - model_pred) ** 2)
# print("神经网络模型的均方误差（MSE）：", model_mse)

# model_pred[model_pred <= 0.6] = 0
# model_pred[(0.6<model_pred) & (model_pred<=1.4)] = 1
# model_pred[model_pred > 1.4] = 2
# for i in model_pred:
#     print(i)

# num = 0
# for i in range(len(model_pred)):
#     if model_pred[i] == test_labels.values[i]:
#         num += 1
#
# model_score = num/len(model_pred)
# print(f'神经网络模型预测正确率为: {model_score}')

n_test_labels = to_categorical(test_labels, num_classes=3)
model_loss, model_accuracy = network_model.evaluate(test_data, n_test_labels)
print("神经网络模型的损失: ", model_loss)
print("神经网络模型准确率: ", model_accuracy)
test_labels = init_test_labels

# 保存模型
# model.save('/path/to/save/neural_network_model.h5')






# 以下为决策树
print('\n'*5)
print('以下为两种不同划分基准(gini和entropy)的决策树: ')
time.sleep(2)

train_data = init_train_data
train_labels = init_train_labels
test_data = init_test_data
test_labels = init_test_labels

# 划分基准为gini

mdh_g = [4, 5, 6, 7, 8]
rs_g = [None, 60, 80]
msl_g = [1, 2, 3, 4]
mss_g = list(range(2, 50))
tr_g1 = []
te_g1 = []
model_g_best = 0
score_g_max = 0
score_g = [0, 0, 0, 0]
# 调参过程
for i1 in mdh_g:
    for i2 in rs_g:
        for i3 in msl_g:
            for i4 in mss_g:
                model_find = DecisionTreeClassifier(criterion="gini",
                                                    max_depth=i1,
                                                    random_state=i2,
                                                    min_samples_leaf=i3,
                                                    min_samples_split=i4)
                model_find = model_find.fit(train_data, train_labels)
                score_train = model_find.score(train_data, train_labels)
                score_test = model_find.score(test_data, test_labels)
                # 记录最优模型对应的参数
                if score_test > score_g_max:
                    model_g_best = model_find
                    score_g_max = score_test
                    score_g[0] = i1
                    score_g[1] = i2
                    score_g[2] = i3
                    score_g[3] = i4


# 划分基准为entropy

mdh_e = [4, 5, 6, 7, 8]
rs_e = [None, 40, 60]
msl_e = [2, 3, 4]
mss_e = list(range(50, 150, 2))
score_e_max = 0
score_e = [0, 0, 0, 0]
model_e_best = 0
tr_e1 = []
te_e1 = []
# 调参过程
for i1 in mdh_e:
    for i2 in rs_e:
        for i3 in msl_e:
            for i4 in mss_e:
                model_find = DecisionTreeClassifier(criterion="entropy",
                                                    max_depth=i1,
                                                    random_state=i2,
                                                    min_samples_leaf=i3,
                                                    min_samples_split=i4)
                model_find = model_find.fit(train_data, train_labels)
                score_train = model_find.score(train_data, train_labels)
                score_test = model_find.score(test_data, test_labels)
                # 记录最优模型对应的参数
                if score_test > score_e_max:
                    model_e_best = model_find
                    score_e_max = score_test
                    score_e[0] = i1
                    score_e[1] = i2
                    score_e[2] = i3
                    score_e[3] = i4


# 决策树模型可视化
# # gini可视化
# g_feature = train_data.columns
# g_class = np.array(train_labels.unique().tolist()).astype(str)  # 包含的类别
#
# # 因决策树深度可能会导致图片看不清，所以只显示第一层和第二层
# tree.plot_tree(model_g_best,
#                feature_names=g_feature,
#                class_names=g_class,
#                max_depth=1,
#                filled=True,
#                rounded=True)
#
# plt.savefig('tree_g.png')
#
#
# entropy可视化
# e_feature = train_data.columns
# e_class = np.array(train_labels.unique().tolist()).astype(str)  # 包含的类别
#
# # 因决策树深度可能会导致图片看不清，所以只显示第一层和第二层
# tree.plot_tree(model_e_best,
#                feature_names=e_feature,
#                class_names=e_class,
#                max_depth=1,
#                filled=True,
#                rounded=True)
#
# plt.savefig('tree_e.png')
# plt.show()

print('对应决策树参数依次为: 1.树的最大深度 2.划分的随机程度 3.树节点的成叶阈值 4.树节点的分裂阈值')

print(f'entropy决策树的准确率为: {score_e_max}')
print('entropy决策树的参数如下: ')
print(score_e)
time.sleep(2)
print('\n'*3)

print(f'gini决策树的准确率为: {score_g_max}')
print('gini决策树的参数如下: ')
print(score_g)
time.sleep(2)


# 决策树模型可视化
# gini可视化放大版
g_feature = train_data.columns
g_class = np.array(train_labels.unique().tolist()).astype(str) # 包含的类别

# 因决策树深度可能会导致图片看不清，所以只显示前三层
plt.figure(figsize=(200, 100))
tree.plot_tree(model_g_best,
                feature_names=g_feature,
                class_names=g_class,
                max_depth=6,
                filled=True,
                rounded=True,
                fontsize=12)
plt.title("gini决策树")
#plt.savefig('tree_g.png')


# entropy可视化放大版
e_feature = train_data.columns
e_class = np.array(train_labels.unique().tolist()).astype(str) # 包含的类别

# 因决策树深度可能会导致图片看不清，所以只显示前三层
plt.figure(figsize=(200, 100))
tree.plot_tree(model_e_best,
                feature_names=e_feature,
                class_names=e_class,
                max_depth=6,
                filled=True,
                rounded=True,
                fontsize=12)
plt.title("entropy决策树")
#plt.savefig('tree_e.png')

plt.show()





# 集成学习方法stack
time.sleep(2)
print('\n'*3)
print('以下为集成学习的stacking方法')
# 定义函数使得keras框架下的神经网络模型与scikit-learn框架兼容
def create_network():
    model = Sequential()
    model.add(InputLayer(input_shape=(train_data.shape[1],)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 创建KerasClassifier对象
network = KerasClassifier(build_fn=create_network, epochs=50, batch_size=32, verbose=0)


# 为了避免输出值维度和类型不匹配问题，我们使用LogisticRegression而不是LinearRegression

# 基础学习器
estimators = [
    ('logistic', LogisticRegression(max_iter=1000)),
    ('knn', KNeighborsClassifier()),
    ('network', network),
    ('decision_tree', DecisionTreeClassifier())
]

# 集成模型Stacking分类器
stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(multi_class='multinomial', solver='saga', random_state=0))

stack_clf.fit(train_data, train_labels)
stack_pred = stack_clf.predict(test_data)

# 计算准确率
stack_accuracy = accuracy_score(test_labels, stack_pred)
print(f"stacking集成模型的准确率为: {stack_accuracy}")




