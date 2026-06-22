import numpy as np

def lax_wendroff(v,delta_t,delta_x,N,f,p):
    #set u as in my written solutions
    u=(v*delta_t)/delta_x
    #build matrix as computed by hand
    A=np.zeros((N,N))
    for i in range(N):
        A[i,i]=1-u**2
    for i in range(N-1):
        A[i+1,i]=0.5*u*(u+1)
        A[i,i+1]=0.5*u*(u-1)
    A[0,N-1]=0.5*u*(u+1)
    A[N-1,0]=0.5*u*(u-1)
    #construct F which is a matrix of the value of f at time t in the rows and position x in the columns
    F=np.zeros((p+1,N))
    for i in range(N):
        F[0,i]=f[i]
    for i in range(p):
        F[i+1,:]=np.matmul(A,F[i,:])

    return F