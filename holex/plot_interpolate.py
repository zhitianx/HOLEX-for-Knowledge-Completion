import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from scipy.optimize import curve_fit

import csv

from matplotlib import rc
rc('mathtext', default='regular')

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=20)

filtered_hits10 = np.zeros(32)
filtered_mean_rank = np.zeros(32)

with open('KBC.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     i = 0
     for row in spamreader:
         i = i + 1
         if i >= 5:
            t = row[0].split(',')
            filtered_hits10[i - 5] = float(t[4])
            filtered_mean_rank[i - 5] = float(t[1])

#print(filtered_mean_rank)

#print(filtered_hits10)

def plot_double_y_axis(xlim,c1,c2,name):
    dc = np.arange(1,xlim)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.plot(dc, c1, '--b', label = 'filtered_mean_rank',marker = 'o')
    ax2 = ax.twinx()
    lns2 = ax2.plot(dc, c2, '-r', label = 'filtered_hits@10',marker = 'o')

    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=1, borderaxespad=0.)

    #ax.grid()
    ax.set_xlabel("number of 0/1 random vectors")
    ax.set_ylabel("filtered_mean_rank")
    ax2.set_ylabel("filtered_hits@10")
    #ax2.set_ylim(0, 35)
    #ax.set_ylim(-20,100)
    plt.tight_layout()
    plt.savefig(name + '.jpg',dpi = 100)
    plt.savefig(name + '.pdf',dpi = 100)

plot_double_y_axis(33,filtered_mean_rank,filtered_hits10,'interpolate_original')

sliding_average_mean_rank = np.zeros(30)
sliding_average_hits10 = np.zeros(30)

for i in range(30):
    sliding_average_hits10[i] = 0
    sliding_average_mean_rank[i] = 0
    for j in range(3):
        sliding_average_hits10[i] += filtered_hits10[i + j]
        sliding_average_mean_rank[i] += filtered_mean_rank[i + j]
    sliding_average_mean_rank[i] /= 3
    sliding_average_hits10[i] /= 3

plot_double_y_axis(31,sliding_average_mean_rank,sliding_average_hits10,'interpolate_sliding_average_3')


sliding_average_mean_rank = np.zeros(28)
sliding_average_hits10 = np.zeros(28)

for i in range(28):
    sliding_average_hits10[i] = 0
    sliding_average_mean_rank[i] = 0
    for j in range(5):
        sliding_average_hits10[i] += filtered_hits10[i + j]
        sliding_average_mean_rank[i] += filtered_mean_rank[i + j]
    sliding_average_mean_rank[i] /= 5
    sliding_average_hits10[i] /= 5

plot_double_y_axis(29,sliding_average_mean_rank,sliding_average_hits10,'interpolate_sliding_average_5')

