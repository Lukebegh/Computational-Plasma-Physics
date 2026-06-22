import numpy as np

def implicit_upwind(v,delta_t,delta_x,N,f,p):
    #build matrix as computed by hand
    A=np.zeros((N,N))
    for i in range(N):
        A[i,i]=1+(v*delta_t)/delta_x
    for i in range(N-1):
        A[i+1,i]=-(v*delta_t)/delta_x
    A[0,N-1]=-(v*delta_t)/delta_x
    #construct F which is a matrix of the value of f at time t in the rows and position x in the columns
    F=np.zeros((p+1,N))
    for i in range(N):
        F[0,i]=f[i]
    for i in range(p):
        F[i+1,:]=np.matmul(np.linalg.inv(A),F[i,:])

    return F