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


'''
max_number_node = 8381214

id_weibo = '3793991852126437'
rt_weibo = []
rt_time = [-1 for i in np.arange(max_number_node+1)]

time_interval= []

switch = 0
num_su = 0

fr = open('information_diffusion_sort_clear_two.csv')
fwone = open('information_one_source.txt','w')
fwtwo = open('information_diffusion_sort_clear_three.csv','w')
for line in fr:
    col = line.strip('\n').split(',')
    if col[0] == id_weibo:
        if switch == 0:
            source = col[1]
            switch = 1
        if rt_time[int(col[2])] == -1:
            rt_weibo.append([col[1],col[2]])
            rt_time[int(col[2])] = int(col[3])
            if rt_time[int(col[1])] == -1:
                rt_time[int(col[1])] = 0
                num_su = num_su + 1
    else:
        for i in np.arange(len(rt_weibo)):
            m = int(rt_weibo[i][0])
            n = int(rt_weibo[i][1])
            if rt_time[m] != 0:
                interval = rt_time[n]-rt_time[m]
                time_interval.append(interval)
            else:
                if rt_weibo[i][0] == source:
                    interval = rt_time[n]-rt_time[m]
                    time_interval.append(interval)
        if num_su == 1:        
            fwone.write(id_weibo+','+source+','+str(len(rt_weibo))+'\n')
            for i in np.arange(len(rt_weibo)):
                fwtwo.write(id_weibo+','+rt_weibo[i][0]+','+rt_weibo[i][1]+','+str(rt_time[int(rt_weibo[i][1])])+'\n')
        num_su = 0
        source = col[1]
        id_weibo = col[0]
        rt_weibo = []
        rt_time = [-1 for i in np.arange(max_number_node+1)]
        
        if rt_time[int(col[2])] == -1:
            rt_weibo.append([col[1],col[2]])
            rt_time[int(col[2])] = int(col[3])
            if rt_time[int(col[1])] == -1:
                rt_time[int(col[1])] = 0
                num_su = num_su + 1

for i in np.arange(len(rt_weibo)):
    m = int(rt_weibo[i][0])
    n = int(rt_weibo[i][1])
    if rt_time[m] != 0:
        interval = rt_time[n]-rt_time[m]
        time_interval.append(interval)
    else:
        if rt_weibo[i][0] == source:
            interval = rt_time[n]-rt_time[m]
            time_interval.append(interval)
if num_su == 1:        
    fwone.write(id_weibo+','+source+','+str(len(rt_weibo))+'\n')
    for i in np.arange(len(rt_weibo)):
        fwtwo.write(id_weibo+','+rt_weibo[i][0]+','+rt_weibo[i][1]+','+str(rt_time[int(rt_weibo[i][1])])+'\n')
fwtwo.close()
fwone.close()
fr.close()

fw = open('interval_time_clear_two.csv','w')
fw.write(str(time_interval))
fw.close()
'''

time_interval = []

fr = open('interval_time_clear_two.csv')
for line in fr:
    col = line.strip('\n').strip('[').strip(']').split(',')
    for i in np.arange(len(col)):
        time_interval.append(int(col[i]))
fr.close()

print(max(time_interval))

for i in np.arange(len(time_interval)):
    time_interval[i] = time_interval[i]/3600

min_interval = min(time_interval)
max_interval = max(time_interval)

point = 9

interval = np.log10(max_interval-min_interval)/point

gap = []
gap.append(0)
for i in np.arange(point+1):
    gap.append(pow(10,i*interval))

gap[-1] = gap[-1] + 1

a,b = np.histogram(time_interval,bins=gap)
for i in np.arange(len(a)):
    a[i] = a[i]/(gap[i+1]-gap[i])

p = []
for i in np.arange(len(a)):
    p.append(a[i]/sum(a))


plt.figure(2)
fig2 = plt.gcf()
fig2.clear()

plt.loglog(b[1:],p,'ro',markersize=2)

fw = open('interval_time_distribution.txt','w')
for i in np.arange(len(p)):
    fw.write(str(b[i+1])+','+str(p[i])+'\n')
fw.close()
