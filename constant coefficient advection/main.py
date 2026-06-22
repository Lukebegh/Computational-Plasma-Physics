import numpy as np
import math
import matplotlib.pyplot as plt
from Explicit_upwind_scheme import explicit_upwind
from Implicit_upwind_scheme import implicit_upwind
from Lax_Wendroff_scheme import lax_wendroff
from Crank_Nicholson_scheme import crank_nicholson
from Pseudospectral_method import pseudospectral

#INPUTS: Please edit this section to use the code as desired
#constant v
v=2
#time domain [a,b]
a=0
b=2*math.pi
#number of points N
N=100
#length L
L=1
#number of time points p - ensure this is high enough such that the method converges
p=160
#define initial condition
def initial_condition(x,a,b,sigma):
    f=np.zeros(N)
    for i in range(N):
        f[i]=(1/(((2*math.pi)**0.5)*sigma))*math.exp(-((x[i]-(a+b)/2)**2)/(2*sigma**2))
    return f
#choose which method to solve with:
#Explicit upwind: 1
#Implicit upwind: 2
#Lax-Wendroff: 3
#Crank-Nicholson: 4
#Pseudosprectral method: 5
method=5
#choose which graph to show
#Mass m: 1
#L2 norm: 2
graph=2




#SETUP
#compute delta_x
delta_x=L/N
#x vector
x=np.zeros(N)
for i in range(N):
    x[i]=i*delta_x
#compute delta t
delta_t=(b-a)/p
#time vector
t=np.zeros(p+1)
for i in range(p+1):
    t[i]=i*delta_t
#sigma
sigma=1
#compute f initial condition
f=initial_condition(x,a,b,sigma)




#PROCEDURE
if method==1:
    F=explicit_upwind(v,delta_t,delta_x,N,f,p)
elif method==2:
    F=implicit_upwind(v,delta_t,delta_x,N,f,p)
elif method==3:
    F=lax_wendroff(v,delta_t,delta_x,N,f,p)
elif method==4:
    F=crank_nicholson(v,delta_t,delta_x,N,f,p)
elif method==5:
    F=pseudospectral(v,delta_t,delta_x,N,f,p,L)



#RESULTS
#construct mass m
m=np.zeros(p+1)
for i in range(p+1):
    for j in range(N):
        m[i]=m[i]+delta_x*F[i,j]
#construct L2 norm
L2=np.zeros(p+1)
for i in range(p+1):
    for j in range(N):
        L2[i]=L2[i]+delta_x*(F[i,j]**2)
#Show desired graph
if graph==1:
    plt.plot(t,m, color='r', label='Mass')
elif graph==2:
    plt.plot(t,L2, color='g', label='L2 norm')

plt.show()