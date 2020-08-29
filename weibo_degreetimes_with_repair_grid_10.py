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

def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    results['polynomial'] = coeffs.tolist()

    p = np.poly1d(coeffs)
    yhat = p(x)            
    ybar = np.sum(y)/len(y) 
    ssreg = np.sum((yhat-ybar)**2) 
    sstot = np.sum((y - ybar)**2) 
    results['determination'] = ssreg / sstot

    return results

gap = [1.0, 3.0, 5.0, 10.0, 20.0, 41.0, 86.0, 179.0, 375.0, 787.0, 2719414.0]

pr = []
fr = open('pr_heatmap_no_repair_grid_10.txt')
for line in fr:
    x = []
    col = line.strip().split()
    for i in np.arange(len(col)):
        x.append(float(col[i]))
    pr.append(x)
fr.close()

pr_repair = []
fr = open('pr_heatmap_with_repair_grid_10.txt')
for line in fr:
    x = []
    col = line.strip().split()
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
            x.append([np.log(gap[i]),np.log(gap[j])])
            y.append(np.log(pr[i][j]))
        if pr_repair[i][j] != 0:
            x_repair.append([np.log(gap[i]),np.log(gap[j])])
            y_repair.append(np.log(pr_repair[i][j]))

x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())

x_repair = sm.add_constant(x_repair)
model = sm.OLS(y_repair, x_repair)
results = model.fit()
print(results.summary())

x = []
y = []
x_repair = []
y_repair = []
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        if pr[i][j] != 0:
            x.append(pow(gap[i],-0.7290)*pow(gap[j],-0.4735))
            y.append(pr[i][j])
        if pr_repair[i][j] != 0:
            x_repair.append(pow(gap[i],-0.0486)*pow(gap[j],-0.4623))
            y_repair.append(pr_repair[i][j])


'''
ax22 = plt.subplot(235)
ax22.clear()
ax22.scatter(x,y,marker='o',c='darkblue',edgecolors='',linewidths=1,s=20,alpha=0.6)
ax22.loglog()

a,b = np.polyfit(np.log(x),np.log(y),1)
xx = np.arange(min(x),max(x), 0.001)
yy = np.power(xx,a)*np.exp(b)
ax22.loglog(xx,yy,linestyle='-',linewidth=2,color='black',alpha=0.5)
r_squared = polyfit(np.log(x),np.log(y),1)
print(r_squared)

ax22.set_xlabel('${k_d}^\\alpha {k_r}^\\beta$',color='k',fontsize=textsize,labelpad=0,rotation=0,alpha=1)
ax22.set_ylabel('$\Lambda$',color='k',fontsize=textsize,labelpad=10,rotation=0,alpha=1)
ax22.text(-0.27,1.10,'f',color='k',fontsize=textsize,fontweight='bold',rotation=0,alpha=1,transform=ax22.transAxes)
ax22.text(0.55, 0.15,'$\\alpha=-0.73(2)$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax22.transAxes)
ax22.text(0.55, 0.05,'$\\beta=-0.47(2)$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax22.transAxes)
ax22.text(0.05, 0.90,'$R^2=0.90$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax22.transAxes)

ax22.tick_params(axis='both',which='major',direction='in',width=1,length=2,top=True,right=True,pad=1.5,labelsize=ticksize)
ax22.tick_params(axis='both',which='minor',left=False,bottom=False,top=False,right=False)
'''


plt.sca(ax22)
ax22.clear()
ax22.scatter(x_repair,y_repair,marker='o',c='darkblue',edgecolors='',linewidths=1,s=50,alpha=0.6)
ax22.loglog()

a,b = np.polyfit(np.log(x_repair),np.log(y_repair),1)
xx = np.arange(min(x_repair),max(x_repair), 0.001)
yy = np.power(xx,a)*np.exp(b)
ax22.loglog(xx,yy,linestyle='-',linewidth=2,color='black',alpha=0.5)
r_squared = polyfit(np.log(x_repair),np.log(y_repair),1)
print(r_squared)

ax22.set_xlabel('${k_d}^\\alpha {k_r}^\\beta$',color='k',fontsize=textsize,labelpad=0,rotation=0,alpha=1)
ax22.set_ylabel('$\Lambda^a$',color='k',fontsize=textsize,labelpad=15,rotation=0,alpha=1)
ax22.text(-0.27,1.10,'e',color='k',fontsize=textsize,fontweight='bold',rotation=0,alpha=1,transform=ax22.transAxes)
ax22.text(0.65, 0.15,'$\\alpha=-0.05(2)$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax22.transAxes)
ax22.text(0.65, 0.05,'$\\beta=-0.46(2)$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax22.transAxes)
ax22.text(0.05, 0.90,'$R^2=0.83$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax22.transAxes)
ax22.text(0.05, 0.80,'$Slope=1.00$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax22.transAxes)

ax22.tick_params(axis='both',which='major',direction='in',width=1,length=2,top=True,right=True,pad=1.5,labelsize=ticksize)
ax22.tick_params(axis='both',which='minor',left=False,bottom=False,top=False,right=False)

ax22.set_ylim(4e-4,1e-1)
ax22.set_xlim(1e-2,3e0)
