import matplotlib.pyplot as plt

a = [1,11,24,43,53,54]
b = [88.88,88.95,88.95,88.95,88.93,89.1]
c = [98.36,98.49,98.62,98.62,98.62,98.74]
# plt.title('adaption bn layer number')
plt.figure()
plt.xticks([1,11,24,43,54])
# shot的分布
plt.plot(a, b, label="A->D", linestyle="--")
# NBNS分布
plt.plot(a, c, label="D->W", linestyle="-")
plt.legend()
# plt.ylim(88.7,89.2)
plt.xlabel('bn layer')
plt.ylabel('accuracy')
# plt.plot(a, b)
plt.show()
# plt.savefig('bn_num.png',dpi=300)