import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# 加载分类数据
iris = datasets.load_iris()
# 在这里只讨论两个特征的情况, 因为多于两个特征是无法进行可视化的
X = iris.data[:, 0:2]
y = iris.target

# 使用SVM分类器
clf = svm.LinearSVC().fit(X, y)
# 接下来进行可视化, 要想进行可视化, 我们核心就是要调用plt.contour函数画图, 但是它要求传入三个矩阵, 而我们的x1和x2为向量, 预测的值也为向量, 所有我们需要将x1和x2转换为矩阵

# 获取边界范围, 为了产生数据
x1_min, x1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
x2_min, x2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

# 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
# 有了新的数据, 我们需要将这些数据输入到分类器获取到结果, 但是因为输入的是矩阵, 我们需要给你将其转换为符合条件的数据
Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
# 这个时候得到的是Z还是一个向量, 将这个向量转为矩阵即可
Z = Z.reshape(xx1.shape)
plt.figure()
# 分解的时候有背景颜色
plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.RdYlBu)
# 为什么需要输入矩阵, 因为等高线函数其实是3D函数, 3D坐标是三个平面, 平面对应矩阵
plt.contour(xx1, xx2, Z, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()
