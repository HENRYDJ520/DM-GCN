import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import time as time
from numpy import random

# def acc(self, x):
#
#     return att

#
# if __name__ == '__main__':
#     low = [1, 1, 1, 1]
#     up = [25, 25, 25, 25]
#     pso = PSO(4, 100, 50, low, up, -1, 1, w=0.9)
#     pso.pso()


"""
普通的四流融合脚本
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ntu120/xsub', help='the work folder for storing results')
# parser.add_argument('--dataset', default='volleyball', help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation', type=float)
parser.add_argument('--joint_dir', default='config/MSCMotion/joint', help='Directory containing "score.pkl" for joint eval results')
parser.add_argument('--bone_dir', default='config/MSCMotion/bone', help='Directory containing "score.pkl" for bone eval results')
parser.add_argument('--joint_motion_dir', default='config/MSCMotion/joint_motion')
parser.add_argument('--bone_motion_dir', default='config/MSCMotion/bone_motion')
parser.add_argument('--angule_dir', default='config/MSCMotion/angule_angule_sigle')
arg = parser.parse_args()

dataset = arg.dataset
if '120' in arg.dataset:
    if 'xsub' in arg.dataset:
        npz_data = np.load('./data/ntu120/NTU120_XSub.npz')
        label = np.where(npz_data['y_test'] > 0)[1]

        index_needed = np.where(((label > 48) & (label < 60)) | (label > 104))
        label = label[index_needed]
        label = np.where(((label > 48) & (label < 60)), label - 49, label)
        label = np.where((label > 104), label - 94, label)

        print("120sub--------------------------------------------------------")
    elif 'xset' in arg.dataset:
        npz_data = np.load('./data/ntu120/NTU120_XSet.npz')
        label = np.where(npz_data['y_test'] > 0)[1]

        index_needed = np.where(((label > 48) & (label < 60)) | (label > 104))
        label = label[index_needed]
        label = np.where(((label > 48) & (label < 60)), label - 49, label)
        label = np.where((label > 104), label - 94, label)
elif 'volley' in arg.dataset:
    npz_data = np.load('./data/volley/train_joint.npz')
    label = np.where(npz_data['y_test'] > 0)[1]

else:
    raise NotImplementedError

with open(os.path.join(arg.joint_dir, 'score.pkl'), 'rb') as r1:
    r1 = list(pickle.load(r1).items())

if arg.bone_dir is not None:
    with open(os.path.join(arg.bone_dir, 'score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

if arg.joint_motion_dir is not None:
    with open(os.path.join(arg.joint_motion_dir, 'score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3).items())
if arg.bone_motion_dir is not None:
    with open(os.path.join(arg.bone_motion_dir, 'score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4).items())
if arg.angule_dir is not None:  # 角度
    with open(os.path.join(arg.angule_dir, 'score.pkl'), 'rb') as r5:
        r5 = list(pickle.load(r5).items())
right_num = total_num = right_num_5 = 0


class PSO:

    def __init__(self, D, N, M, p_low, p_up, v_low, v_high, w=1., c1=2., c2=2.):
        self.w = w  # 惯性权值
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 群体学习因子
        self.D = D  # 粒子维度
        self.N = N  # 粒子群规模，初始化种群个数
        self.M = M  # 最大迭代次数
        self.p_range = [p_low, p_up]  # 粒子位置的约束范围
        self.v_range = [v_low, v_high]  # 粒子速度的约束范围
        self.x = np.zeros((self.N, self.D))  # 所有粒子的位置
        self.v = np.zeros((self.N, self.D))  # 所有粒子的速度
        self.p_best = np.zeros((self.N, self.D))  # 每个粒子的最优位置
        self.g_best = np.zeros((1, self.D))[0]  # 种群（全局）的最优位置
        self.p_bestFit = np.zeros(self.N)  # 每个粒子的最优适应值
        self.g_bestFit = float('Inf')  # float('-Inf')，始化种群（全局）的最优适应值，由于求极小值，故初始值给大，向下收敛，这里默认优化问题中只有一个全局最优解

        # 初始化所有个体和全局信息
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = random.uniform(self.p_range[0][j], self.p_range[1][j])
                self.v[i][j] = random.uniform(self.v_range[0], self.v_range[1])
            self.p_best[i] = self.x[i]  # 保存个体历史最优位置，初始默认第0代为最优
            fit = self.fitness(self.p_best[i])
            self.p_bestFit[i] = fit  # 保存个体历史最优适应值
            if fit < self.g_bestFit:  # 寻找并保存全局最优位置和适应值
                self.g_best = self.p_best[i]
                self.g_bestFit = fit

    def pso(self, draw=1):
        best_fit = []  # 记录每轮迭代的最佳适应度，用于绘图
        w_range = None
        if isinstance(self.w, tuple):
            w_range = self.w[1] - self.w[0]
            self.w = self.w[1]
        time_start = time.time()  # 记录迭代寻优开始时间
        for i in range(self.M):
            self.update()  # 更新主要参数和信息
            if w_range:
                self.w -= w_range / self.M  # 惯性权重线性递减
            print("\rIter: {:d}/{:d} fitness: {:.4f} ".format(i, self.M, -self.g_bestFit, end='\n'))
            g_bestFit = self.g_bestFit
            best_fit.append(g_bestFit)
        time_end = time.time()  # 记录迭代寻优结束时间
        print(f'Algorithm takes {time_end - time_start} seconds')  # 打印算法总运行时间，单位为秒/s
        if draw:
            plt.figure()
            plt.plot([i for i in range(self.M)], best_fit)
            plt.xlabel("iter")
            plt.ylabel("fitness")
            plt.title("Iter process")
            plt.show()

    def update(self):
        for i in range(self.N):
            # 更新速度
            self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + self.c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.D):
                if self.v[i][j] < self.v_range[0]:
                    self.v[i][j] = self.v_range[0]
                if self.v[i][j] > self.v_range[1]:
                    self.v[i][j] = self.v_range[1]
            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.D):
                if self.x[i][j] < self.p_range[0][j]:
                    self.x[i][j] = self.p_range[0][j]
                if self.x[i][j] > self.p_range[1][j]:
                    self.x[i][j] = self.p_range[1][j]
            # 更新个体和全局历史最优位置及适应值
            _fit = self.fitness(self.x[i])
            # 个体位置信息
            if _fit < self.p_bestFit[i]:
                self.p_best[i] = self.x[i]
                self.p_bestFit[i] = _fit
            if _fit < self.g_bestFit:
                self.g_best = self.x[i].copy()
                self.g_bestFit = _fit

    def fitness(self, x):
        """
        根据粒子位置计算适应值
        """
        right_num = total_num = right_num_5 = 0
        # arg.alpha = [0.6, 0.6, 0.3, 0.2, 0.5]
        # 0.6, 0.6, 0.6, 0.2, 0.6 是MSCAMotion的参数
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * x[0] + r22 * x[1] + r33 * x[2]
            # print(x[0], x[1], x[2])
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        return -acc


if __name__ == '__main__':
    low = [0, 0, 0]
    up = [1, 1, 1]
    pso = PSO(3, 100, 65, low, up, -1, 1, w=0.9)
    pso.pso()
# python enu_two_graph.py --dataset 120/xsub --joint_dir two_person_score/hypergnn --bone_dir two_person_score/two_graph --joint_motion_dir two_person_score/alltransformer
