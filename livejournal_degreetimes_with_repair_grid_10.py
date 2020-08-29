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

gap_i = [1.0, 6.0, 10.0, 17.0, 29.0, 50.0, 87.0, 152.0, 265.0, 463.0, 70001.0]
gap_j = [1.0, 17.0, 29.0, 50.0, 87.0, 152.0, 265.0, 463.0, 808.0, 1411.0, 70001.0]

pr = []
fr = open('pr_heatmap_with_repair_grid_10_bin_0-3.txt')
for line in fr:
    x = []
    col = line.strip().split()
    for i in np.arange(len(col)):
        x.append(float(col[i]))
    pr.append(x)
fr.close()

x = []
y = []
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        if pr[i][j] != 0:
            x.append([np.log(gap_i[i]),np.log(gap_j[j])])
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

x = []
y = []
for i in np.arange(len(pr)):
    for j in np.arange(len(pr[i])):
        if pr[i][j] != 0:
            x.append(pow(gap_i[i],-1.2212)*pow(gap_j[j],0.2906))
            y.append(pr[i][j])

plt.sca(ax21)
ax21.clear()
ax21.scatter(x,y,marker='o',c='darkblue',edgecolors='',linewidths=1,s=50,alpha=0.6)
ax21.loglog()

a,b = np.polyfit(np.log(x),np.log(y),1)
xx = np.arange(min(x),max(x), 0.00001)
yy = np.power(xx,a)*np.exp(b)
ax21.loglog(xx,yy,linestyle='-',linewidth=2,color='black',alpha=0.5)
r_squared = polyfit(np.log(x),np.log(y),1)
print(r_squared)

ax21.set_xlabel('${k_d}^\\alpha {k_r}^\\beta$',color='k',fontsize=textsize,labelpad=0,rotation=0,alpha=1)
ax21.set_ylabel('$\Lambda^a$',color='k',fontsize=textsize,labelpad=15,rotation=0,alpha=1)
ax21.text(-0.27,1.10,'d',color='k',fontsize=textsize,fontweight='bold',rotation=0,alpha=1,transform=ax21.transAxes)
ax21.text(0.65, 0.15,'$\\alpha=-1.22(2)$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax21.transAxes)
ax21.text(0.65, 0.05,'$\\beta=\ \ \ \ 0.29(2)$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax21.transAxes)
ax21.text(0.05, 0.90,'$R^2=0.94$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax21.transAxes)
ax21.text(0.05, 0.80,'$Slope=1.00$',fontsize=ticksize,color='k',fontweight='normal',rotation=0,alpha=1,transform=ax21.transAxes)

ax21.tick_params(axis='both',which='major',direction='in',width=1,length=2,top=True,right=True,pad=1.5,labelsize=ticksize)
ax21.tick_params(axis='both',which='minor',left=False,bottom=False,top=False,right=False)

ax21.set_xlim(4e-4,9.9e0)
