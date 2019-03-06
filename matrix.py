# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:02:35 2019

@author: 老四
"""

import numpy as np
import copy

def f(A,x):
    B=A.I
    Kh=B*x
    K=Kh.T*Kh
    B_=B.T*B
    F=np.mat(np.zeros([3,3]))
    for i in [0,1,2]:
        for j in [0,1,2]:
            M=np.mat([[-1*B_[a,i]*B[j,b] for a in [0,1,2]] for b in [0,1,2]])
            F[i,j]=x.T*B.T*M*x
    return K,F/K

A1=np.mat(np.random.rand(3,3))
x=np.mat([[1],[2],[3]])


i,j=1,2
a=0.01


B1=A1.I
C1=B1*x
K1=C1.T*C1
T1=np.sqrt(K1)

print('T1:\n',T1)
M1=np.mat([[B1[a,i]*B1[j,b] for b in [0,1,2]] for a in [0,1,2]])
M2=M1*x
M3=C1.T*M2
M4=M3/T1

A2=copy.deepcopy(A1)
A2[i,j]+=a
B2=A2.I
C2=B2*x
K2=C2.T*C2
T2=np.sqrt(K2)

T3=T1-M4*a
print('T2:\n',T2)
print('T3:\n',T3)
print('T2-T3:\n',T2-T3)

"""
K1,M=f(A,x)


A[i,j]+=a
K_=K1+a*M[i,j]
K,_=f(A,x)
print(K1,K,K_,K-K_)
"""