import numpy as np

def crank_nicholson(v,delta_t,delta_x,N,f,p):
    #set u as in my written solutions
    u=(v*delta_t)/delta_x
    #build matrices as computed by hand
    #A
    A=np.zeros((N,N))
    for i in range(N):
        A[i,i]=1
    for i in range(N-1):
        A[i+1,i]=-0.25*u
        A[i,i+1]=0.25*u
    A[0,N-1]=-0.25*u
    A[N-1,0]=0.25*u
    #B
    B=np.zeros((N,N))
    for i in range(N):
        B[i,i]=1
    for i in range(N-1):
        B[i+1,i]=0.25*u
        B[i,i+1]=-0.25*u
    B[0,N-1]=0.25*u
    B[N-1,0]=-0.25*u
    #compute A inverse times B
    A_inv=np.linalg.inv(A)
    C=np.matmul(A_inv,B)
    #construct F which is a matrix of the value of f at time t in the rows and position x in the columns
    F=np.zeros((p+1,N))
    for i in range(N):
        F[0,i]=f[i]
    for i in range(p):
        F[i+1,:]=np.matmul(C,F[i,:])

    return F