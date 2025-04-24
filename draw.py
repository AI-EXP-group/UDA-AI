import matplotlib.pyplot as plt
# allX=[1, 2, 3, 4, 5, 6,7,8,9,10]
# allY=[117, 131, 141, 93, 101, 62]
# stuX=[1, 2, 3, 4, 5, 6]
# adsX=[1, 2, 3, 4, 5, 6]
# adsY=[40, 52, 66, 41, 54, 37]
# nY=[2.04,2.09,1.84,1.72,1.97,2.11,1.89,1.52,1.36,1.26]
# pY=[-0.8,-0.98,-0.87,-1.09,-0.86,-1.20,-0.91,-1.06,-1.0,-0.8]
#
# npY=[2.54,1.09,1.94,1.02,1.56,2.0,1.23,2.52,1.6,1.5]
# ppY=[0.2,0.08,-0.01,0.08,-0.96,-0.6,-0.96,-1.15,-0.3,-0.1]
#
# # 总体度的分布
# plt.figure()
# plt.plot(allX, nY, label="wrong", linestyle=":")
#
# plt.plot(allX, npY, label="pre wrong", linestyle="-.")
#
# plt.plot(allX, ppY, label="pre correct", linestyle="-")
#
# # # advisee度的折线图分布
# # plt.plot(stuX, stuY,label="advisee度的分布", linestyle="--")
#
# # advisor度的折线图分布
# plt.plot(allX, pY, label="correct", linestyle="-.")
# plt.legend()
# plt.title("simlarity")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()
plt.figure()
X=[1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
shot=[71.8,72.1,73.2,73.5,73.8, 73.9,73.8,74.0,74.2,74.1,74.3,74.5,74.6,74.6,74.55,74.45,74.55,74.6 ,74.58,74.56]
our=[70.71,72.63,73.3,73.87,74.33,74.8,74.8,75,75.19,75.36,75.68,76,76.24,76.39,76.5,76.58,76.55,76.6,76.58,76.6]

plt.plot(X, shot, label="SHOT", linestyle="--")
plt.plot(X, our, label="Our", linestyle="-.")
plt.title("D->A")
plt.xlabel("epoch")
# plt.xlim(0, 20)
plt.xticks(range(21))
plt.ylabel("accuracy")
plt.legend()
# plt.show()
plt.savefig('accuracy.png', dpi=300)