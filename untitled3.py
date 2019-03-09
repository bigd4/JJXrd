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
import ase.io
import StructuretoXRD as stx

import sdtw
from sdtw.distance import SquaredEuclidean

def f(a,b):
    D = SquaredEuclidean(a, b)
    dtw=sdtw.SoftDTW(D, gamma=0.001)
    err=dtw.compute()
    E = dtw.grad()
    derr_dI = D.jacobian_product(E).reshape(-1)
    return err#,derr_dI


lamb=1.5
thetarange=[10,18]
plotargs={'function':'Lorentzian','w':0.1,'step':0.05}

a=ase.io.read('3.vasp')
b=ase.io.read('2.vasp')
fa=np.array(stx.getplot(a,lamb,thetarange,plotargs)).transpose(1,0)
fb=np.array(stx.getplot(b,lamb,thetarange,plotargs)).transpose(1,0)
fc=copy.deepcopy(fb)
for c in fc:
    c[1]=0

fa_=np.array(stx.getplot(a,lamb,thetarange,plotargs)[0]).reshape(-1,1)
fb_=np.array(stx.getplot(b,lamb,thetarange,plotargs)[0]).reshape(-1,1)

