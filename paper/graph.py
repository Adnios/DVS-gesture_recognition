import os
import numpy as np
import matplotlib.pyplot as plt

# 可视化为了画折线图
step_list1 = [615,1105,1401,1521,2116,2127,2187,3335,4540,5199]   # tra_acc
step_list2 = [2536,3793,823,2444,1940,2821,1662,2559,2264,960]   # tra_acc
cnn_list1 = [0.9918,1,0.7259,0.8902,0.9524,0.9312,0.877,0.9045,0.8653,0.8609]   # tra_acc
cnn_list2 = [1,2,3,4,5,6,7,8,9,10]   # tra_acc

fig = plt.figure()  # 建立可视化图像框
ax = fig.add_subplot(2, 1, 1)  # 子图总行数、列数，位置
ax.yaxis.grid(True)
ax.set_title('cnn_accuracy ', fontsize=14, y=1.02)
ax.set_xlabel('step')
ax.set_ylabel('accuracy')
bx = fig.add_subplot(2, 1, 2)
bx.yaxis.grid(True)
bx.set_title('cnn_loss ', fontsize=14, y=1.02)
bx.set_xlabel('step')
bx.set_ylabel('loss')


ax.plot(step_list1, cnn_list1, color="r", label='Line1')
bx.plot(step_list2, cnn_list2, color="r", label='Line2')

plt.tight_layout()
plt.show()
