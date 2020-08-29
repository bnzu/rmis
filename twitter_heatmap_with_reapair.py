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
import statsmodels.api      as sm

from sklearn.linear_model import LinearRegression


max_number_node = 456626

node_to_degree = [0 for i in np.arange(max_number_node+1)]
node_to_neighbour = [[] for i in np.arange(max_number_node+1)]

fr = open('higgs-social_network_edgelist.txt')
for line in fr:
    col = line.strip('\n').split(' ')
    node_to_degree[int(col[0])] = node_to_degree[int(col[0])] + 1
    node_to_degree[int(col[1])] = node_to_degree[int(col[1])] + 1
    node_to_neighbour[int(col[1])].append(col[0])
fr.close()

print('section one')

node_to_transmit = [[] for i in np.arange(max_number_node+1)]

fr = open('higgs-activity_time_retweet_second_clear.txt')
for line in fr:
    col = line.strip('\n').split(' ')
    node_to_transmit[int(col[1])].append(col[0])
fr.close()

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

numgrid = len(gap)-1

rt_edge = np.zeros([numgrid,numgrid])
sn_edge = np.zeros([numgrid,numgrid])
sn_repeat = []

for u in np.arange(len(node_to_transmit)):

    if node_to_transmit[u] != []:
        for v in np.arange(len(node_to_transmit[u])):
            w = node_to_transmit[u][v]

            for i in np.arange(len(gap)):
                if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
                    break
            for j in np.arange(len(gap)):
                if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
                    break

            rt_edge[i][j] = rt_edge[i][j] + 1

        for v in np.arange(len(node_to_neighbour[u])):
            w = node_to_neighbour[u][v]
            
            for i in np.arange(len(gap)):
                if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
                    break
            for j in np.arange(len(gap)):
                if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
                    break

            sn_edge[i][j] = sn_edge[i][j] + 1
            sn_repeat.append(str(w)+'-'+str(u))


'''
sn_norepeat = set(sn_repeat)
sn_norepeat = list(sn_norepeat)

print('section two')

num_edge = 0
obser = np.zeros([numgrid,numgrid])
for k in np.arange(len(sn_norepeat)):
    num_edge = num_edge + 1
    if num_edge % 100000 == 0:
        print(num_edge)
    
    col = sn_norepeat[k].split('-')
    kd = len(node_to_neighbour[int(col[1])])
    kr = node_to_degree[int(col[0])]-len(node_to_neighbour[int(col[0])])

    for i in np.arange(len(gap)):
        if gap[i] <= kd <= gap[i+1]:
            break
    for j in np.arange(len(gap)):
        if gap[j] <= kr <= gap[j+1]:
            break

    obser[i][j] = obser[i][j] + 1

print('section three')

num_edge = 0
graph = np.zeros([numgrid,numgrid])
fr = open('higgs-social_network_edgelist.txt')
for line in fr:
    num_edge = num_edge + 1
    if num_edge % 1000000 == 0:
        print(num_edge)
    col = line.strip('\n').split(' ')
    w = int(col[0])
    u = int(col[1])
    for i in np.arange(len(gap)):
        if gap[i] <= len(node_to_neighbour[int(u)]) <= gap[i+1]:
            break
    for j in np.arange(len(gap)):
        if gap[j] <= node_to_degree[int(w)]-len(node_to_neighbour[int(w)]) <= gap[j+1]:
            break
    graph[i][j] = graph[i][j] + 1
fr.close()

print('section four')

pr_edge = np.zeros([numgrid,numgrid])
pr_edge_repair = np.zeros([numgrid,numgrid])
for i in np.arange(numgrid):
    for j in np.arange(numgrid):
        if sn_edge[i][j] > 0 and graph[i][j] > 0:
            pr_edge[i][j] = rt_edge[i][j]/sn_edge[i][j]
            pr_edge_repair[i][j] = rt_edge[i][j]/sn_edge[i][j]*obser[i][j]/graph[i][j]
            
pr = []
for i in np.arange(len(pr_edge)):
    pr.append(list(pr_edge[i]))

pr_repair = []
for i in np.arange(len(pr_edge_repair)):
    pr_repair.append(list(pr_edge_repair[i]))
'''


fw = open('rt_heatmap_no_repair_grid_10.txt','w')
for i in np.arange(len(rt_edge)):
    for j in np.arange(len(rt_edge[i])):
        fw.write(str(rt_edge[i][j])+' ')
    fw.write('\n')
fw.close()

fw = open('sn_heatmap_no_repair_grid_10.txt','w')
for i in np.arange(len(sn_edge)):
    for j in np.arange(len(sn_edge[i])):
        fw.write(str(sn_edge[i][j])+' ')
    fw.write('\n')
fw.close()


'''
fw = open('pr_heatmap_no_repair_grid_10.txt','w')
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        fw.write(str(pr[i][j])+' ')
    fw.write('\n')
fw.close()

fw = open('pr_heatmap_with_repair_grid_10.txt','w')
for i in np.arange(len(pr_repair)):
    for j in np.arange(len(pr_repair[i])):
        fw.write(str(pr_repair[i][j])+' ')
    fw.write('\n')
fw.close()
'''


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


'''
fig1 = plt.figure(2)
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

plt.savefig('pr_heatmap_grid_10.png')
'''
