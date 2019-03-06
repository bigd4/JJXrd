# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:45:28 2019

@author: 老四
"""

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
    K=np.sqrt(Kh.T*Kh)
    print(Kh)
    F=np.mat(np.zeros([3,3]))
    for i in [0,1,2]:
        for j in [0,1,2]:
            M=np.mat([[B[a,i]*B[j,b] for b in [0,1,2]] for a in [0,1,2]])
            F[i,j]=Kh.T*M*x
    return K,F/K

A1=np.mat(np.random.rand(3,3))
x=np.mat([[1],[2],[3]])


i,j=0,2
a=0.01

K1,F1=f(A1,x)

A2=copy.deepcopy(A1)
A2[i,j]+=a
K2,_=f(A2,x)

K3=K1-F1[i,j]*a

print('K1:\n',K1)
print('K2:\n',K2)
print('K3:\n',K3)
print('K2-K3:\n',K2-K3)