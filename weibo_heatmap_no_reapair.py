import gc
import sys
import time
import random
import string
import matplotlib

import numpy                as np
import scipy                as sp
import matplotlib.pyplot    as plt
import networkx             as nx
import statsmodels.api      as sm

from sklearn.linear_model import LinearRegression


max_number_node = 8381214

node_to_degree = [0 for i in np.arange(max_number_node+1)]
#node_to_neighbour = [[] for i in np.arange(max_number_node+1)]

l = 0
fr = open('user_relation_directed.csv')
for line in fr:
    col = line.strip('\n').split(',')
    node_to_degree[int(col[0])] = node_to_degree[int(col[0])] + 1
    node_to_degree[int(col[1])] = node_to_degree[int(col[1])] + 1
    #node_to_neighbour[int(col[1])].append(int(col[0]))
    
    l = l + 1
    if l % 100000000 == 0:
        print(l)
fr.close()

print('sectionone')

max_degree = max(node_to_degree)

grid = 20

interval = np.log10(max_degree)/grid

gaptem = []

for i in np.arange(grid+1):
    gaptem.append(np.ceil(pow(10,i*interval)))

choosepoint = [0,1,2,3,4,5,6,7,8,9,20]
gap = []

for i in choosepoint:
    gap.append(gaptem[i])


'''
numgrid = 10

rt_edge = np.zeros([numgrid,numgrid])
sn_edge = np.zeros([numgrid,numgrid])

count_post = 1

fr = open('information_diffusion_sort_clear_two_split_sort.txt')
for line in fr:
    col = line.strip('\n').split('\t')
    num_post = col[0]
    diss = col[1]
    break
fr.close()

fr = open('information_diffusion_sort_clear_two_split_sort.txt')
for line in fr:
    col = line.strip('\n').split('\t')
    if col[0] == num_post:
        if col[1] == diss:
            u = int(col[1])
            w = int(col[2])
            for i in np.arange(len(gap)):
                if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
                    break
            for j in np.arange(len(gap)):
                if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
                    break

            rt_edge[i][j] = rt_edge[i][j] + 1
        else:
            for v in np.arange(len(node_to_neighbour[u])):
                w = int(node_to_neighbour[u][v])
                for i in np.arange(len(gap)):
                    if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
                        break
                for j in np.arange(len(gap)):
                    if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
                        break
                
                sn_edge[i][j] = sn_edge[i][j] + 1
            
            diss = col[1]
            u = int(col[1])
            w = int(col[2])
            for i in np.arange(len(gap)):
                if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
                    break
            for j in np.arange(len(gap)):
                if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
                    break

            rt_edge[i][j] = rt_edge[i][j] + 1
    else:
        for v in np.arange(len(node_to_neighbour[u])):
            w = int(node_to_neighbour[u][v])
            for i in np.arange(len(gap)):
                if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
                    break
            for j in np.arange(len(gap)):
                if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
                    break

            sn_edge[i][j] = sn_edge[i][j] + 1
        
        count_post = count_post + 1
        if count_post % 1000 == 0:
            print(count_post)

        num_post = col[0]
        diss = col[1]
        u = int(col[1])
        w = int(col[2])
        for i in np.arange(len(gap)):
            if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
                break
        for j in np.arange(len(gap)):
            if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
                break

        rt_edge[i][j] = rt_edge[i][j] + 1
fr.close()

for v in np.arange(len(node_to_neighbour[u])):
    w = int(node_to_neighbour[u][v])
    for i in np.arange(len(gap)):
        if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
            break
    for j in np.arange(len(gap)):
        if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
            break

    sn_edge[i][j] = sn_edge[i][j] + 1
fr.close()

print('sectiontwo')

all_dissem = []

fr = open('information_diffusion_sort_clear_two_split_sort.txt')
for line in fr:
    col = line.strip('\n').split('\t')
    all_dissem.append(int(col[1]))
fr.close()

all_dissem_norepeat = set(all_dissem)
all_dissem_norepeat = list(all_dissem_norepeat)

print(len(all_dissem))
print(len(all_dissem_norepeat))

obser = np.zeros([numgrid,numgrid])
for k in all_dissem_norepeat:
    for v in node_to_neighbour[k]:
        for i in np.arange(len(gap)):
            if gap[i] <= len(node_to_neighbour[k]) <= gap[i+1]:
                break
        for j in np.arange(len(gap)):
            if gap[j] <= node_to_degree[int(v)]-len(node_to_neighbour[int(v)]) <= gap[j+1]:
                break
                
        obser[i][j] = obser[i][j] + 1

print('sectionthree')


num_edge = 0
graph = np.zeros([numgrid,numgrid])
fr = open('user_relation_directed.csv')
for line in fr:
    num_edge = num_edge + 1
    if num_edge % 10000000 == 0:
        print(num_edge)
    col = line.strip('\n').split(',')
    w = int(col[0])
    u = int(col[1])
    for i in np.arange(len(gap)):
        if gap[i] <=len(node_to_neighbour[int(u)]) <= gap[i+1]:
            break
    for j in np.arange(len(gap)):
        if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
            break
    graph[i][j] = graph[i][j] + 1
fr.close()

print('sectionfour')

pr_edge = np.zeros([numgrid,numgrid])
pr_edge_repair = np.zeros([numgrid,numgrid])
for i in np.arange(numgrid):
    for j in np.arange(numgrid):
        if sn_edge[i][j] > 0 and graph[i][j] > 0:
            pr_edge_repair[i][j] = rt_edge[i][j]/sn_edge[i][j]*obser[i][j]/graph[i][j]
            pr_edge[i][j] = rt_edge[i][j]/sn_edge[i][j]

pr = []
for i in np.arange(len(pr_edge)):
    pr.append(list(pr_edge[i]))

pr_repair = []
for i in np.arange(len(pr_edge_repair)):
    pr_repair.append(list(pr_edge_repair[i]))

fw = open('pr_grid_10.txt','w')
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        fw.write(str(pr[i][j])+' ')
    fw.write('\n')
fw.close()

fw = open('pr_repair_grid_10.txt','w')
for i in np.arange(len(pr_repair)):
    for j in np.arange(len(pr_repair[i])):
        fw.write(str(pr_repair[i][j])+' ')
    fw.write('\n')
fw.close()
'''


pr = []
fr = open('pr_heatmap_no_repair_grid_10.txt')
for line in fr:
    x = []
    col = line.strip('\n').strip(' ').split(' ')
    for i in np.arange(len(col)):
        x.append(float(col[i]))
    pr.append(x)
fr.close()

pr_repair = []
fr = open('pr_heatmap_with_repair_grid_10.txt')
for line in fr:
    x = []
    col = line.strip('\n').strip(' ').split(' ')
    for i in np.arange(len(col)):
        x.append(float(col[i]))
    pr_repair.append(x)
fr.close()

x = []
y = []
x_repair = []
y_repair = []
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        if pr[i][j] != 0:
            x.append([np.log(gap[i+1]),np.log(gap[j+1])])
            y.append(np.log(pr[i][j]))
        if pr_repair[i][j] != 0:
            x_repair.append([np.log(gap[i+1]),np.log(gap[j+1])])
            y_repair.append(np.log(pr_repair[i][j]))

model = LinearRegression()

model.fit(x,y)
print(model.coef_)
print(model.intercept_)
print('R-squared: %.2f' % model.score(x, y))

model_repair = LinearRegression()

model_repair.fit(x_repair,y_repair)
print(model_repair.coef_)
print(model_repair.intercept_)
print('R-squared: %.2f' % model_repair.score(x_repair, y_repair))

x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())

x_repair = sm.add_constant(x_repair)
model = sm.OLS(y_repair, x_repair)
results = model.fit()
print(results.summary())


'''
fig1 = plt.figure(1)
ax11 = plt.subplot(121)
ax11.clear()
cm = plt.cm.rainbow
sc = ax11.imshow(pr, cmap = cm, vmin=min(min(pr)), vmax=max(max(pr)),alpha=1)

cbar = fig1.add_axes([0.05, 0.17, 0.02, 0.64])
fig1.colorbar(sc, cax=cbar)
cbar.tick_params(direction='out',width=0.5,length=4,left=False,right=True,colors='k',pad=2,labelsize=12,labelcolor='k',labelrotation=0,labelleft=False,labelright=True)

ax12 = plt.subplot(122)
ax12.clear()
cm = plt.cm.rainbow
sc = ax12.imshow(pr_repair, cmap = cm, vmin=min(min(pr_repair)), vmax=max(max(pr_repair)),alpha=1)

cbar = fig1.add_axes([0.95, 0.17, 0.02, 0.64])
fig1.colorbar(sc, cax=cbar)
cbar.tick_params(direction='out',width=0.5,length=4,left=False,right=True,colors='k',pad=2,labelsize=12,labelcolor='k',labelrotation=0,labelleft=False,labelright=True)

plt.savefig('pr_heatmap.png')
'''
