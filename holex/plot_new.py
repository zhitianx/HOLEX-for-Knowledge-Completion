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


with open('nips.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     i = 0
     for row in spamreader:
         i = i + 1
         #print(row)

def plot_double_y_axis(xlim,c1,c2,name):
    dc = np.arange(1,xlim)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.plot(dc, c2, '--b', label = 'filtered_mean_rank',marker='o')
    ax2 = ax.twinx()
    lns2 = ax2.plot(dc, c1, '-r', label = 'filtered_hits@10',marker='o')

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


def plotlinechart(y,name,sv,c):
    dc = np.arange(1, 9)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # lns1 = ax.plot(dc, filtered_hits1, '-b', label='filtered_hits@1')

    # lns2 = ax.plot(dc, filtered_hits5, '-r', label='filtered_hits@5')

    ax.plot(dc, y, c, label=name,marker= 'o')

    # added these three lines
    # lns = lns1 + lns2 + lns3
    #ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=1, borderaxespad=0.)

    # ax.grid()
    ax.set_xlabel("number of 0/1 random vectors")
    ax.set_ylabel(name)
    # ax2.set_ylim(0, 35)
    # ax.set_ylim(-20,100)
    plt.tight_layout()
    plt.savefig(sv + '.jpg', dpi=100)
    plt.savefig(sv + '.pdf', dpi=100)


hits10 = np.array([0.82888,0.8538115149565777,0.865433122852161,0.8707402955765097,0.8745831287772342,0.8777657395337815,0.877613380508202,0.878857645883767])
mean_rank = np.array([55.6875,48.10233447884749,45.148787052868585,46.58342503089503,47.77175771529177,47.20218889133416,45.925792690152534,46.62067681264919])
time_each_epoch = np.array([760,857,894,934,963,1091,1185,1277])

plot_double_y_axis(9,hits10,mean_rank,'hits10_meanrank')

plotlinechart(time_each_epoch,'time per epoch(secs)','timeperiteration','-g')




