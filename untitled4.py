# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:57:19 2019

@author: 老四
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(1,10,0.01)
I=10
w=0.1
for m in [1,2,3,4,5]:
    y=I*w**(2*m)/(w**2+(2**(1/m)-1)*(x-5)**2)**m
    plt.plot(x,y)


y=I*w**2/(w**2+(x-5)**2)