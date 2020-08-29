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

fr = open('information_diffusion_tree_second_clear.txt')
for line in fr:
    col = line.strip('\n').split(',')
    number_node[0] = max(int(col[1]),int(col[2]))
    number_node[1] = max(number_node[0],number_node[1])
fr.close()

print('section one')

max_number_node = number_node[1]

rt_time = [-1 for i in np.arange(max_number_node+1)]

time_interval= []

switch = 0
num_su = 0

fr = open('information_diffusion_tree_second_clear.txt')
for line in fr:
    col = line.strip('\n').split(',')
    if switch == 0:
            id_weibo = col[0]
            switch = 1
    if col[0] == id_weibo:
        rt_time[int(col[2])] = int(col[3])
        if rt_time[int(col[1])] == -1:
            rt_time[int(col[1])] = 0
            num_su = num_su + 1
        interval = rt_time[int(col[2])]-rt_time[int(col[1])]
        time_interval.append(interval)
    else:
        if num_su != 1:        
            print(num_su)

        num_su = 0
        id_weibo = col[0]
        rt_time = [-1 for i in np.arange(max_number_node+1)]
        
        rt_time[int(col[2])] = int(col[3])
        rt_time[int(col[1])] = 0
        num_su = num_su + 1
        interval = rt_time[int(col[2])]-rt_time[int(col[1])]
        time_interval.append(interval)
        
if num_su != 1:        
    print(num_su)
fr.close()

print('section two')

fw = open('interval_time_second_clear.csv','w')
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
'''


print(max(time_interval))

for i in np.arange(len(time_interval)):
    time_interval[i] = time_interval[i]/3600

min_interval = min(time_interval)
max_interval = max(time_interval)

point = 20

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

plt.figure(1)
fig1 = plt.gcf()
fig1.clear()

plt.loglog(b[1:],p,'ro',markersize=2)

fw = open('interval_time_distribution.csv','w')
for i in np.arange(len(p)):
    fw.write(str(b[i+1])+'\t'+str(p[i])+'\n')
fw.close()
