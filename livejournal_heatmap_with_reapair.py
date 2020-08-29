import gc
import sys
import time
import string
import matplotlib

import random
import numpy                as np
import scipy                as sp
import matplotlib.pyplot    as plt
import networkx             as nx
import statsmodels.api      as sm

from sklearn.linear_model import LinearRegression


'''
max_number_node = 9573126

node_to_degree = [0 for i in np.arange(max_number_node+1)]
node_to_neighbour = [[] for i in np.arange(max_number_node+1)]

l = 0
fr = open('social_network_livejournal.txt')
for line in fr:
    col = line.strip('\n').split(',')
    node_to_degree[int(col[0])] = node_to_degree[int(col[0])] + 1
    node_to_degree[int(col[1])] = node_to_degree[int(col[1])] + 1
    node_to_neighbour[int(col[0])].append(int(col[1]))
    node_to_neighbour[int(col[1])].append(int(col[0]))
    
    l = l + 1
    if l % 10000000 == 0:
        print(l)
fr.close()

print('sectionone')

max_degree = max(node_to_degree)

grid = 20

interval = np.log10(max_degree)/grid

gap= []

for i in np.arange(grid+1):
    gap.append(np.ceil(pow(10,i*interval)))

gap_i = []
gap_j = []

gap_i.append(gap[0])
for i in np.arange(3,12):
    gap_i.append(gap[i])
gap_i.append(gap[20])

gap_j.append(gap[0])
for j in np.arange(5,14):
    gap_j.append(gap[j])
gap_j.append(gap[20])

numgrid = 10

rt_edge = np.zeros([numgrid,numgrid])

count_post = 1

fr = open('information_diffusion_tree_third_clear_sort.txt')
for line in fr:
    col = line.strip('\n').split('\t')
    num_post = col[0]
    diss = col[1]
    break
fr.close()

fr = open('information_diffusion_tree_third_clear_sort.txt')
for line in fr:
    col = line.strip('\n').split('\t')
    if col[0] == num_post:
        if col[1] == diss:
            u = int(col[1])
            w = int(col[2])
            for i in np.arange(len(gap_i)):
                if gap_i[i] <= node_to_degree[int(u)] <= gap_i[i+1]:
                    break
            for j in np.arange(len(gap_j)):
                if gap_j[j] <= node_to_degree[int(w)] <= gap_j[j+1]:
                    break

            rt_edge[i][j] = rt_edge[i][j] + 1
        else:
            diss = col[1]
            u = int(col[1])
            w = int(col[2])
            for i in np.arange(len(gap_i)):
                if gap_i[i] <= node_to_degree[int(u)] <= gap_i[i+1]:
                    break
            for j in np.arange(len(gap_j)):
                if gap_j[j] <= node_to_degree[int(w)] <= gap_j[j+1]:
                    break

            rt_edge[i][j] = rt_edge[i][j] + 1
    else:
        count_post = count_post + 1
        if count_post % 10000 == 0:
            print(count_post)

        num_post = col[0]
        diss = col[1]
        u = int(col[1])
        w = int(col[2])
        for i in np.arange(len(gap_i)):
            if gap_i[i] <= node_to_degree[int(u)] <= gap_i[i+1]:
                break
        for j in np.arange(len(gap_j)):
            if gap_j[j] <= node_to_degree[int(w)] <= gap_j[j+1]:
                break

        rt_edge[i][j] = rt_edge[i][j] + 1
fr.close()

print('sectiontwo')

totalpost = 0
num_post = [0 for i in np.arange(max_number_node+1)]

fr = open('userpostnum.txt')
for line in fr:
    col = line.strip('\n').split(' ')
    num_post[int(col[0])] = int(col[1])
    totalpost =totalpost + int(col[1])
fr.close()

print('totalpost',totalpost)

spreader = []
fr = open('information_diffusion_tree_third_clear_sort.txt')
for line in fr:
    col = line.strip('\n').split('\t')
    spreader.append(int(col[1]))
fr.close()

spreader_norepeat = set(spreader)
spreader_norepeat = list(spreader_norepeat)

totalpost_repair = 0
sn_edge = np.zeros([numgrid,numgrid])

for u in spreader_norepeat:
    if u % 10000000 == 0:
        print(u)
    if node_to_degree[u] != 0 and num_post[u] != 0:
        for v in np.arange(len(node_to_neighbour[u])):
            w = int(node_to_neighbour[u][v])
            for i in np.arange(len(gap_i)):
                if gap_i[i] <= node_to_degree[int(u)] <= gap_i[i+1]:
                    break
            for j in np.arange(len(gap_j)):
                if gap_j[j] <= node_to_degree[int(w)] <= gap_j[j+1]:
                    break

            sn_edge[i][j] = sn_edge[i][j] + num_post[u]

        totalpost_repair = totalpost_repair + num_post[u]

print('sectionthree')

print('totalpost_repair',totalpost_repair)

pr_edge = np.zeros([numgrid,numgrid])
for i in np.arange(numgrid):
    for j in np.arange(numgrid):
        if sn_edge[i][j] > 0:
            pr_edge[i][j] = rt_edge[i][j]/sn_edge[i][j]
            
pr = []
for i in np.arange(len(pr_edge)):
    pr.append(list(pr_edge[i]))

fw = open('pr_heatmap_with_repair_grid_10_bin_0-3.txt','w')
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        fw.write(str(pr[i][j])+' ')
    fw.write('\n')
fw.close()
'''

pr = []
fr = open('pr_heatmap_with_repair_grid_10_bin_0-3.txt')
for line in fr:
    x = []
    col = line.strip('\n').strip(' ').split(' ')
    for i in np.arange(len(col)):
        x.append(float(col[i]))
    pr.append(x)
fr.close()

x = []
y = []
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        if pr[i][j] != 0:
            x.append([np.log(gap[i]),np.log(gap[j])])
            y.append(np.log(pr[i][j]))

model = LinearRegression()

model.fit(x,y)
print(model.coef_)
print(model.intercept_)
print('R-squared: %.2f' % model.score(x, y))

x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())


'''
fig1 = plt.figure(1)
ax11 = plt.subplot(111)
ax11.clear()
cm = plt.cm.rainbow
sc = ax11.imshow(pr, cmap = cm, vmin=min(min(pr)), vmax=max(max(pr)),alpha=1)

cbar = fig1.add_axes([0.10, 0.17, 0.02, 0.64])
fig1.colorbar(sc, cax=cbar)
cbar.tick_params(direction='out',width=0.5,length=4,left=False,right=True,colors='k',pad=2,labelsize=12,labelcolor='k',labelrotation=0,labelleft=False,labelright=True)

plt.savefig('pr_heatmap_with_repair_grid_10_bin_0-3.png')
'''
