import numpy as np

def pseudospectral(v,delta_t,delta_x,N,f,p,L):
    #construct F which is a matrix of the value of f at time t in the rows and position x in the columns
    F=np.zeros((p+1,N))
    for i in range(N):
        F[0,i]=f[i]
    #Fourier transform
    f_hat=np.fft.fft(f)
    #set k modes
    k=np.fft.fftfreq(N,L/N)
    #iterate to find the f_hats and record their inverse transforms
    for i in range(p):
        f_hat=np.exp(-1j*2*np.pi*k*v*delta_t/L)*f_hat
        F[i+1,:]=np.fft.ifft(f_hat).real

    return F