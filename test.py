import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

iris = load_iris()
# take the first two features
X = iris.data[:, :2]
y = iris.target
# train the model
clf = LinearSVC().fit(X, y)
# get the range of the picture
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# get the points to predict
# len(np.arange(x_min, x_max, 0.02)) = 280
# len(np.arange(y_min, y_max, 0.02)) = 220
# xx.shape = (220L, 280L)
# yy.shape = (220L, 280L)
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
# predict the point
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z.shape = (220L, 280L)
Z = Z.reshape(xx.shape)
plt.figure()
# red yellow blue
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# red yellow green
#plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlGn)
# red yellow blue
#from matplotlib.colors import ListedColormap
#plt.contourf(xx, yy, Z, cmap=ListedColormap(['r', 'y', 'b']))
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['r', 'y', 'b']))
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with linear kernel')
plt.show()

# import numpy as np
# # import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn import svm
#
# # 加载分类数据
# iris = datasets.load_iris()
# # 在这里只讨论两个特征的情况, 因为多于两个特征是无法进行可视化的
# X = iris.data[:, 0:2]
# y = iris.target
#
# # 使用SVM分类器
# clf = svm.LinearSVC().fit(X, y)
# # 接下来进行可视化, 要想进行可视化, 我们核心就是要调用plt.contour函数画图, 但是它要求传入三个矩阵, 而我们的x1和x2为向量, 预测的值也为向量, 所有我们需要将x1和x2转换为矩阵
#
# # 获取边界范围, 为了产生数据
# x1_min, x1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
# x2_min, x2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
#
# # 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
# xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
# # 有了新的数据, 我们需要将这些数据输入到分类器获取到结果, 但是因为输入的是矩阵, 我们需要给你将其转换为符合条件的数据
# Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
# # 这个时候得到的是Z还是一个向量, 将这个向量转为矩阵即可
# Z = Z.reshape(xx1.shape)
# plt.figure()
# # 分解的时候有背景颜色
# plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.RdYlBu)
# # 为什么需要输入矩阵, 因为等高线函数其实是3D函数, 3D坐标是三个平面, 平面对应矩阵
# plt.contour(xx1, xx2, Z, cmap=plt.cm.RdYlBu)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.array([0,0,1,1,2,3,3,2,4,5,6,7,8,9])
# y = np.array([0,0,1,2,2,3,3,0,4,5,2,7,5,9])
# xx, yy = np.meshgrid(x, y)
# # print(yy)
#
# z = xx == yy
# # print(z)
# plt.contourf(xx, yy, z, cmap="cool")
# # print(z)
# plt.scatter(xx, yy, c=z)
# plt.show()


# import network
# import torch
# import torch.nn as nn
#
# netF = network.ResBase(res_name="resnet50")
# netB = network.feat_bootleneck(type="bn", feature_dim=netF.in_features)
# netC = network.feat_classifier(type="wn", class_num=31)
# # print(netC)
# for k, v in netC.named_parameters():
#     # a = v[0]
#     print(k, v)
#     print(v.shape)
# cnt = 0
# for module in netF.modules():
#     print(module)
#     # if isinstance(module, nn.BatchNorm2d):
#     #     cnt += 1
#     #     if cnt == 1:
#     #         break
#
# print(cnt)