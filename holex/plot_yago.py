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

filtered_hits10 = np.zeros(5)
filtered_hits5 = np.zeros(5)
filtered_hits1 = np.zeros(5)

with open('yago.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     i = 0
     for row in spamreader:
         i = i + 1
         if i >= 2 and i <= 6:
             t = row[0].split(',')
             filtered_hits1[i - 2] = float(t[4])
             filtered_hits5[i - 2] = float(t[8])
             filtered_hits10[i - 2] = float(t[5])

print(filtered_hits1)

print(filtered_hits5)

print(filtered_hits10)

dc = np.arange(1, 6)

fig = plt.figure()
ax = fig.add_subplot(111)

#lns1 = ax.plot(dc, filtered_hits1, '-b', label='filtered_hits@1')

#lns2 = ax.plot(dc, filtered_hits5, '-r', label='filtered_hits@5')

lns3 = ax.plot(dc, filtered_hits10, '-b', label = 'filtered_hits@10')

# added these three lines
#lns = lns1 + lns2 + lns3
lns = lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=1, borderaxespad=0.)

#ax.grid()
ax.set_xlabel("number of 0/1 random vectors")
#ax.set_ylabel("filtered_mean_rank")
# ax2.set_ylim(0, 35)
# ax.set_ylim(-20,100)
plt.tight_layout()
plt.savefig('yago' + '.jpg', dpi=100)
plt.savefig('yago' + '.pdf', dpi=100)




