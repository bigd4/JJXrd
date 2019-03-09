import ase.io
import ase
import numpy as np
import StructuretoXRD as stx

import sdtw
from sdtw.distance import SquaredEuclidean

#np.set_printoptions(precision=3,suppress=True)
"""
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
"""

######################################################################################   
#1d-dtw
#def GD(a,Ib):
#    Ib=Ib.reshape(-1,1)
#    anglea,Ia,dI_dpositions,dI_dcell=stx.getplot(a,lamb,thetarange,plotargs,True)
#    Ia=Ia.reshape(-1,1)
#    D = SquaredEuclidean(Ia, Ib)
#    dtw=sdtw.SoftDTW(D, gamma=0.01)
#    err=dtw.compute()
#    E = dtw.grad()
#    derr_dI = D.jacobian_product(E).reshape(-1)
#    dcell=np.sum(derr_dI.reshape(-1,1,1)*dI_dcell,axis=0)
#    dpositions=np.sum(derr_dI.reshape(-1,1,1)*dI_dpositions,axis=0)
#    dcell/=(np.max(dcell)+1)*10
#    dpositions/=(np.max(dpositions)+1)*10
#    return err,dcell,dpositions
######################################################################################    

###################################################################################### 
#2d-dtw
#def GD(a,Ib):
#    Ib=Ib.reshape(-1,1)
#    anglea,Ia,dI_dpositions,dI_dcell=stx.getplot(a,lamb,thetarange,plotargs,True)
#    Ia=Ia.reshape(-1,1)
#    D = SquaredEuclidean(Ia, Ib)
#    dtw=sdtw.SoftDTW(D, gamma=0.01)
#    err=dtw.compute()
#    E = dtw.grad()
#    derr_dI = D.jacobian_product(E).reshape(-1)
#    dcell=np.sum(derr_dI.reshape(-1,1,1)*dI_dcell,axis=0)
#    dpositions=np.sum(derr_dI.reshape(-1,1,1)*dI_dpositions,axis=0)
#    dcell/=(np.max(dcell)+1)*10
#    dpositions/=(np.max(dpositions)+1)*10
#    return err,dcell,dpositions
###################################################################################### 

def compare(a,b,lamb=1.5,thetarange=[10,20],sigma=0.05,step=0.01): 
    import matplotlib.pyplot as plt
    [x1,y1]=stx.getplot(a,lamb,thetarange,plotargs)
    [x2,y2]=stx.getplot(b,lamb,thetarange,plotargs)
    plt.figure()
    plt.plot(x1,y1-0.5)
    plt.plot(x2,y2+0.5)
    plt.show()
    
def similarpeak(a,peakb):
    peaka=stx.getpeak(a,lamb,thetarange).transpose(1,0)[0]
    similar=True
    for pb in peakb:
        similar=similar and np.sum(np.abs(peaka-pb)<0.5)>0
    return similar

lamb=1.5
thetarange=[10,20]
plotargs={'function':'Lorentzian','w':0.1,'step':0.05}

a=ase.io.read('3.vasp')
b=ase.io.read('2.vasp')

"""
Ia=stx.getplot(a,lamb,thetarange,plotargs)[1]
Ib=stx.getplot(b,lamb,thetarange,plotargs)[1]
peakb=stx.getpeak(b,lamb,thetarange).transpose(1,0)[0]

step=50
e=1e-5
e=-5
alpha=0.1

for i in range(step):
    err,dcell,dpositions=GD(a,Ib)
    if err<e:
        break
    #if not similarpeak(a,peakb):
    #a.set_scaled_positions(a.get_scaled_positions()-dpositions*alpha)
    _=a.get_scaled_positions()
    a.set_cell(a.get_cell()+dcell*alpha)
    a.set_scaled_positions(_)
    
    print('err:',err)
    if i%5==0:
        print('err:',err)
        print('similarity:',similarpeak(a,peakb))
        print('i:',i)
        compare(a,b,lamb,thetarange,sigma=1)
        ase.io.write('out{}.cif'.format(i),a)
ase.io.write('out.cif'.format(i),a)
compare(a,b,lamb,thetarange,sigma=1)
"""
"""
a=ase.io.read('2.vasp')
[xrda, der]=stx.getpeak(a,lamb,thetarange,True)
_,dh_dcell,dtheta_dcell=der

import copy

alpha=0.1
i,j=0,0
M=np.zeros([3,3])
M[i,j]=1

c=copy.deepcopy(a)
_=c.get_scaled_positions()
c.set_cell(c.get_cell()+alpha*M)
c.set_scaled_positions(_)

xrdc=stx.getpeak(c,lamb,thetarange)
peaka=xrda.transpose(1,0)[0]
peakc=xrdc.transpose(1,0)[0]

peakc_=peaka+alpha*dtheta_dcell[:,i,j]

print('a:{}\nc:{}\nc_:{}'.format(peaka,peakc,peakc_))
"""
import copy

alpha=0.1
i,j=0,0
M=np.zeros([3,3])
M[i,j]=1

a=ase.io.read('2.vasp')
xrda=stx.XrdStructure(a,lamb,thetarange,True)
pa=xrda.peaks[0]

c=copy.deepcopy(a)
_=c.get_scaled_positions()
c.set_cell(c.get_cell()+alpha*M)
c.set_scaled_positions(_)

xrdc=stx.XrdStructure(c,lamb,thetarange,True)
pc=xrdc.peaks[0]

der=pa.dKh_dcell[i,j]*alpha

print('a:{}\nc:{}\nc_:{}'.format(pa.Kh,pc.Kh,pa.Kh+der))