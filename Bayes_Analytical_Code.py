#import pylab
#mport sys
#import numpy as np 
#from scipy import *
#import matplotlib as plt
#import numpy as np
#ys.path.append('/usr/lib/python2.7/dist-packages/numpy')
# Implement linespace in order to change grid spaces
import numpy as np
from math import *
from random import *
#import matplotlib.pyplot as plt

def Init_Distribution(nx, ny):
    global X
    global Y
    X = np.random.random(400)*nx
    Y = np.random.random(400)*ny
    X = np.append(X, np.random.normal(loc=800, scale = 20, size = 300))
    Y = np.append(Y, np.random.normal(loc=600, scale = 80, size = 300))
    X = np.append(X, np.random.normal(loc= 200, scale = 50, size = 100))
    Y = np.append(Y, np.random.normal(loc= 300, scale = 50, size = 100))
    #figure(num=None, figsize=(9.5,9))
    #xlabel('$\mathrm{x}$', fontsize = 35)
    #ylabel('$\mathrm{y}$', fontsize = 35)
    #title('$\mathrm{Random}$ $\mathrm{Cartesian}$ $\mathrm{Distribution}$ ', fontsize = 40)
    #tick_params(axis='both', which='major', labelsize=18)
    #scatter(X, Y)
    #ylim([0, ny])
    #xlim([0, nx])
    #savefig('RandomCartesianDistribution.png')
    return X, Y

def Init_Distribution_Polar(nR):
    global R
    global theta
    R = np.random.random(1000)*nR
    theta = (np.random.random(1000)+1)*randrange(0, 360)
    #figure(num=None, figsize=(9.5,9))
    #xlabel('$\mathrm{x}$', fontsize = 35)
    #ylabel('$\mathrm{y}$', fontsize = 35)
    #title('$\mathrm{Random}$ $\mathrm{Polar}$ $\mathrm{Distribution}$', fontsize = 40)
    #tick_params(axis='both', which='major', labelsize=18)
    #scatter(R*cos(theta), R*sin(theta))
    #xlim([-120, 120])
    #ylim([-120, 120])
    #savefig('RandomPolarDistribution.png')
    return R, theta

# k = Number of Neighbors, D = Dimension of the space we assume here X = Y | ++POLAR COORDINATES++ 
def Neighbours_Polar(k, D):
    global d4Polar
    d4Polar = []
    for i in range(-D, D):# Be carefull with this -D this aplies to the Polar case or where the inital Distribution have negative points
        for j in range(-D, D):
            d = sqrt((R*cos(theta)-i)**2 + (R*sin(theta)-j)**2)# check this equation 
            d2 = sorted(d)
            d3 = d2[0:k]
            d4Polar.append(d3)
    #return d4Polar
    
# k = Number of Neighbors, D = Dimension of the space we assume here X = Y | CARTESIAN COORDINATES
def Neighbours_Cartesian(k, D):
    global d4
    d4 = []
    for i in range(D):
        for j in range(D):
            d = np.sqrt((X-i)**2 + (Y-j)**2)
            d2 = sorted(d)
            d3 = d2[0:k]
            d4.append(d3) 

#Solution,put this into a function
def solution(K, D):
    k = range(1, 100)
    global T3
    Ndots = D*D
    T3 = []
    for i in range (Ndots): #escala los puntos del espacio
        T = []
        for j in range (K):# j escala los vecinos
            teo = d4[i][j]**2 / k[j] # + (d4[i][1]**2 / k[1])) 
            T.append(teo)
        T2 = (sqrt(sum(T))) 
        T3.append(1/(T2*T2*pi)) #This is divided in order to get n_0
	print T3[0:10]

# Check color this creates the grid in order to make the density plot
def plots(D):
    f = open('CartesianDensityData.txt', 'w')
    x = []
    y = []
    for i in range(D):
        for j in range(D):
            x.append(i)
            y.append(j)
    for i in range(len(x)):
        f.write(str(x[i]) + "  " + str(y[i]) + "\n") #+ str(T3[i]) + "\n")
    f.close()
    print len(x), len(y), len(T3)
    #print y[0:100]

Init_Distribution(100, 100)
Neighbours_Cartesian(12, 100)
solution(12, 100)
plots(100)
     
