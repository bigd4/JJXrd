import numpy as np
import StructuretoXRD as stx

def distance(x,y):
    return (x[0]-y[0])**2+abs(x[1]-y[1])


def dtw(X,Y):
    l1,l2=len(X),len(Y) 
    M=np.array([[distance(X[i],Y[j]) for j in range(l2)] for i in range(l1)])
    D=np.zeros([l1+1,l2+1]) 
    
    D[0,:]=np.inf
    D[:,0]=np.inf
    D[0,0]=0
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            D[i][j]=M[i-1][j-1]+min(D[i-1][j],D[i][j-1],D[i-1][j-1]+M[i-1][j-1])
    return D[-1,-1]

def compare_(a,b,lamb,thetarange,sigma=0.05,step=0.01):
    [_,x1]=stx.getplot(a,lamb,thetarange,sigma,step)
    [_,x2]=stx.getplot(b,lamb,thetarange,sigma,step)
    return dtw(x1,x2)

def compare(a,b,lamb,thetarange):
    x=stx.getpeak(a,lamb,thetarange)
    y=stx.getpeak(b,lamb,thetarange)
    return dtw(x,y)
