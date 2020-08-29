import gc
import sys
import time
import string
import community
import matplotlib

import numpy                as np
import scipy                as sp
import matplotlib.pyplot    as plt
import networkx             as nx

number_node = [0,0]

fr = open('higgs-social_network_edgelist.txt')
for line in fr:
    col = line.strip('\n').split(' ')
    number_node[0] = max(int(col[0]),int(col[1]))
    number_node[1] = max(number_node[0],number_node[1])
fr.close()

max_number_node = number_node[1]

time_interval= []

rt_time = [-1 for i in np.arange(max_number_node+1)]

fr = open('higgs-activity_time_retweet_second_clear.txt')
for line in fr:
    col = line.strip('\n').split(' ')
    if rt_time[int(col[0])] == -1:
        rt_time[int(col[0])] = int(col[2])
        if rt_time[int(col[1])] == -1:
            rt_time[int(col[1])] = 0
        if rt_time[int(col[1])] != 0:
            interval = rt_time[int(col[0])]-rt_time[int(col[1])]
            time_interval.append(interval/3600)
fr.close()

min_interval = min(time_interval)
max_interval = max(time_interval)

point = 10

interval = np.log10(max_interval-min_interval)/point

gap = []
gap.append(0)
for i in np.arange(point+1):
    gap.append(pow(10,i*interval))

gap[-1] = gap[-1] + 1
a,b = np.histogram(time_interval,bins=gap)

for i in np.arange(len(a)):
    a[i] = a[i]/(gap[i+1]-gap[i])

t = []
p = []

for i in np.arange(len(a)):
    if a[i] != 0 and b[i+1] != 0:
        t.append(b[i+1])
        p.append(a[i]/sum(a))

fw = open('interval_time_distribution.csv','w')
for i in np.arange(len(p)):
    fw.write(str(t[i])+'\t'+str(p[i])+'\n')
fw.close()

'''
plt.loglog(t,p,'ro',markersize=6)

a,b = np.polyfit(np.log(t),np.log(p),1)
x = np.arange(1,95)
y = np.power(x,a)*np.exp(b)
plt.plot(x,y,'k-',linewidth=2)
plt.text(0.97,0.85,'(c)',color='k',fontsize=textsize,transform=ax31.transAxes)
plt.xlabel('$\\tau$',fontsize=textsize)
plt.ylabel('$P$',fontsize=textsize)
'''

