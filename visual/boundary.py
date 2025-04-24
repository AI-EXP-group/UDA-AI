import torch
import torch.nn.modules
import torch.nn
import numpy as np
from torch.autograd import Variable  # torch的基本变量
import torch.nn.functional as F  # 里面有很多torch的函数
import matplotlib.pyplot as plt


# 定义自带forward propagation的神经网络。
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hiddens, n_outputs):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hiddens)
        self.predict = torch.nn.Linear(n_hiddens, n_outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        predict = F.softmax(self.predict(x))
        return predict


class MyNet:
    def __init__(self, n_features, n_hiddens, n_outputs, times):
        self.NeuronalNet = Net(n_features, n_hiddens, n_outputs)
        self.realX = None
        self.realY = None
        self.opitimizer = None
        self.lossFunc = None
        self.times = times

    # 训练集
    def getData(self):
        temp = torch.ones(100, 2)

        B = torch.normal(2 * temp, 1)

        By = torch.ones(100)
        A = torch.normal(-2 * temp, 1)
        Ay = torch.zeros(100)

        self.realX = Variable(torch.cat([A, B], 0))
        self.realY = Variable(torch.cat([Ay, By]).type(torch.LongTensor))

        # plt.scatter(realX.data.numpy()[:,0],realX.data.numpy()[:,1],c=realY)
        # plt.show()

    def run(self):
        self.opitimizer = torch.optim.SGD(self.NeuronalNet.parameters(), lr=0.01)
        self.lossFunc = torch.nn.CrossEntropyLoss()

        for i in range(self.times):
            out = self.NeuronalNet(self.realX)

            loss = self.lossFunc(out, self.realY)

            self.opitimizer.zero_grad()

            loss.backward()

            self.opitimizer.step()

    # 可视化
    def showBoundary(self):
        x_min, x_max = self.realX[:, 0].min() - 0.1, self.realX[:, 0].max() + 0.1
        y_min, y_max = self.realX[:, 1].min() - 0.1, self.realX[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
        cmap = plt.cm.Spectral

        X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        y_pred = self.NeuronalNet(X_test)
        _, y_pred = y_pred.max(dim=1)
        y_pred = y_pred.reshape(xx.shape)

        plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(self.realX[:, 0], self.realX[:, 1], c=self.realY, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("binary classifier")
        plt.show()

    def predict(self, inputData):
        # inputData should be a 1x2 matrix
        data = torch.from_numpy(np.array(inputData)).int()
        return self.NeuronalNet(data.float())


if __name__ == "__main__":
    myNet = MyNet(2, 18, 2, 1000)
    myNet.getData()
    myNet.run()
    myNet.showBoundary()
    probabilitys = list(myNet.predict([3, 3]).data.numpy())
    print("是第{0}类".format(1 + probabilitys.index(max(probabilitys))))



# 这里我稍微调整了下plt.contour中的参数，使得结果更好看一点
def plot_decision_boundary(model, x, y):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    h = 0.01
    # 绘制网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 生成与网格上所有点对应的分类结果
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # 绘制contour
    plt.contour(xx, yy, z, levels=[0.5], colors=['blue'])
    plt.scatter(x[:, 0], x[:, 1], c=y)



