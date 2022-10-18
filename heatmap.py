#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# vegetables = ["Layer1", "Layer2", "Layer3", "Layer4",
#               "Layer5", "Layer6"]
vegetables = ["I", "swim", "across", "the", "river", "."]
#蔬菜类
# farmers = ["H1", "H2", "H3","H4"]
farmers = ["I", "swim", "across", "the", "river", "."]
#农民类

# harvest = np.array([[0.7327, 0.0128, 0.0354, 0.6245],
#         [0.1551, 0.4441, 0.5109, 0.8390],
#         [0.2123, 0.2514, 0.7737, 0.5988],
#         [0.3732, 0.7197, 0.6897, 0.8174],
#         [0.7645, 0.8196, 0.5680, 0.7830],
#         [0.7752, 0.9834, 0.9136, 0.8890]])
harvest = np.array([[1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]])

#定义热力图数据

fig, ax = plt.subplots()
#将元组分解为fig和ax两个变量
#im = ax.imshow(harvest)
im = ax.imshow(harvest,vmin=0,vmax=1)
fig.colorbar(im,ax=ax)
#显示图片


ax.set_xticks(np.arange(len(farmers)))
#设置x轴刻度间隔
ax.set_yticks(np.arange(len(vegetables)))
#设置y轴刻度间隔
ax.set_xticklabels(farmers)
#设置x轴标签'''
ax.set_yticklabels(vegetables)
#设置y轴标签'''

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
#设置标签 旋转45度 ha有三个选择：right,center,left（对齐方式）


# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")
'''
画图
j,i:表示坐标值上的值
harvest[i, j]表示内容
ha有三个选择：right,center,left（对齐方式）
va有四个选择：'top', 'bottom', 'center', 'baseline'（对齐方式）
color:设置颜色
'''

#ax.set_title("alpha value different layers")
#设置题目
fig.tight_layout()  #自动调整子图参数,使之填充整个图像区域。
plt.savefig("heatmap.jpg",dpi=600)
plt.show()      #图像展示
