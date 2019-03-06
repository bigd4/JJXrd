import ase.io
import ase
import numpy as np
import StructuretoXRD as stx

import sdtw
from sdtw.distance import SquaredEuclidean

#np.set_printoptions(precision=3,suppress=True)
def GD(a,xrdb):
    [xrda, der]=stx.getpeak(a,lamb,thetarange,True)
    #print(der)
    dh_dpositions,dh_dcell,dtheta_dcell=der
    D = SquaredEuclidean(xrda, xrdb)
    dtw=sdtw.SoftDTW(D, gamma=1.0)
    err=dtw.compute()
    E = dtw.grad()
    G = D.jacobian_product(E).transpose(1,0)
    derr_dtheta,derr_dh=G
    dcell=np.sum(derr_dtheta.reshape(-1,1,1)*dtheta_dcell,axis=0)+np.sum(derr_dh.reshape(-1,1,1)*dh_dcell,axis=0)
    dpositions=np.sum(derr_dh.reshape(-1,1,1)*dh_dpositions,axis=0)
    dcell/=(np.max(dcell)+1)*10
    dpositions/=(np.max(dpositions)+1)*10
    return err,dcell,dpositions

def compare(a,b,lamb=1.5,thetarange=[10,40],sigma=0.05,step=0.01): 
    import matplotlib.pyplot as plt
    [x1,y1]=stx.getplot(a,lamb,thetarange,sigma,step)
    [x2,y2]=stx.getplot(b,lamb,thetarange,sigma,step)
    plt.plot(x1,y1-0.5)
    plt.plot(x2,y2+0.5)
    
    
lamb=1.5
thetarange=[10,20]

a=ase.io.read('1.vasp')
b=ase.io.read('2.vasp')
xrda=stx.getpeak(a,lamb,thetarange)
xrdb=stx.getpeak(b,lamb,thetarange)

step=1000
e=1e-3
alpha=0.01

for i in range(step):
    err,dcell,dpositions=GD(a,xrdb)
    if err<e:
        break
    #a.set_cell(a.get_cell()-dcell*alpha)
    a.set_scaled_positions(a.get_scaled_positions()-dpositions*alpha)
    #if i%1000==0:
    print(err)
ase.io.write('out.cif',a)

#compare(a,b,lamb,thetarange,sigma=1)
