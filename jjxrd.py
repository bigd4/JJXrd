# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:00:42 2018

@author: 老四
"""
import ase
import ase.io
import numpy as np
from atomic_form_factors import ff
import matplotlib.pyplot as plt
import itertools
from evaluate import getEMD,overlap,comparepeak

angle_=np.load('N.npz')['angle']
I_=np.load('N.npz')['I']
angles_=np.array([8.049320,8.181150,9.045410,11.140140,7.316890,11.374510,11.843260,4.906223,5.770673])
Is_=np.array([306.584200,290.514510,206.252920,160.128270,131.165200,56.098770,79.021790,42.823206,25.218805])/1297.807671

def compress(P,n=20,l=None):
    """
    (x,y)=(np.arange(n),np.sum(np.concatenate((P[1],np.zeros(n-len(P[1])%n))).reshape(-1,len(P[1])//n+1),1))
    plt.figure()
    plt.scatter(x,y)
    return (x,y)
    """
    if l is not None:
        P=(P[0][l],P[1][l])
    return (np.arange(n),np.sum(np.concatenate((P[1],np.zeros(n-len(P[1])%n))).reshape(-1,len(P[1])//n+1),1))
    
class XRD():
    def __init__(self,hkl,theta,Kh,atoms):
        self.hkl=hkl
        self.multi=1
        self.theta=theta
        self.Kh=Kh
        self.d=2*np.pi/Kh
        self.atoms=atoms
        self.positions=atoms.get_scaled_positions()

    def get_f(self,symbol):
        f = ff[symbol]
        f0=f[-1]
        for i in np.arange(0, 8, 2):
            f0 += f[i] *np.exp(-f[i+1]*(self.Kh/(4.*np.pi))**2) * np.exp(-0.01 * self.Kh**2 /(4.0*np.pi))
        return f0

    def get_F(self):
        self.F=0
        for i,atom in enumerate(self.atoms):
            self.F+=self.get_f(atom.symbol)*np.exp(2.0*np.pi*1j*np.dot(self.hkl,self.positions[i]))
        return abs(self.F)**2

    def get_I(self):
        LP = 1/np.sin(self.theta)**2/np.cos(self.theta)
        P = 1 + np.cos(2*self.theta)**2
        self.I = self.get_F()*LP*P*self.multi
        return self.I
        #return self.get_F()
        
class XrdStructure():
    def __init__(self,atoms,lamb,name='QAQ',theta=[0,90]):
        self.atoms=atoms
        self.lattice=self.atoms.get_cell_lengths_and_angles()
        self.reciprocal_lattice=self.atoms.get_reciprocal_cell()*2*np.pi
        self.lamb=lamb
        self.name=name
        [self.thetamin,self.thetamax]=theta
        self.Khmax=4*np.pi*np.sin(self.thetamax/180*np.pi)/lamb
        self.Khmin=4*np.pi*np.sin(self.thetamin/180*np.pi)/lamb
        self.peaks=[]
        
        self.getallhkl()
        self.angles=np.array([peak.theta/np.pi*360 for peak in self.peaks])
        self.Is=np.array([peak.get_I() for peak in self.peaks])
        self.Is=self.Is/np.sum(self.Is)
        #for peak in self.peaks:
        #    peak.get_I()
        #    self.angles.append(peak.theta/np.pi*360)
        #    self.Is.append(peak.I)
        #self.Is=self.Is/np.sum(self.Is)
        self.evaluate()
    
    def xiajibahua(self):
        plt.figure()
        angle=np.arange(2*self.thetamin,2*self.thetamax,0.01)
        I=np.array([self.f(x,big=False) for x in angle])
        plt.plot(angle,I,label=self.name)
        plt.plot(angle_,I_,label='shiyan')
        #plt.text(self.thetamin+0.5,np.max(I)-0.5,self.evaluate(),fontsize=15)
        plt.plot([4],[0],label=self.evaluate)
        plt.legend()
        
    def getallhkl(self):
        index_range = np.arange(16, -17,-1)
        for hkl in itertools.product(index_range, repeat=3):
            theta=self.gettheta(hkl)
            if theta:
                for peak in self.peaks: 
                    if np.allclose(theta,peak.theta):
                        peak.multi+=1
                        theta=False
                        break
                if theta:
                    self.peaks.append(XRD(hkl,theta,self.getKh(hkl),self.atoms))
            
    def gettheta(self,hkl):
        if self.getKh(hkl)<self.Khmin or self.getKh(hkl)>self.Khmax:
            return False
        else:
            return np.arcsin(self.getKh(hkl)*self.lamb/4/np.pi)

    def getKh(self,hkl):
        Kh=np.dot(hkl,self.reciprocal_lattice)
        return np.sqrt(np.dot(Kh,Kh))

    def f(self,x,sigma=0.05,big=True):
        f=0
        for h,mu in zip(self.Is,self.angles):
            if h==max(self.Is) and big:
                break
            f+=h/sigma/np.sqrt(2*np.pi)*np.e**(-0.5*(x-mu)**2/sigma**2)
        return f
    
    def evaluate(self,method='EMD'):
        P=(angles_,Is_)
        Q=(self.angles,self.Is)
        if comparepeak(P,Q):
            if method=='overlap':
                P=(angle_,I_)
                Q=(angle_,np.array([self.f(x) for x in angle_]))
                self.evaluate=2*overlap(compress(P,20),compress(Q,20))+overlap(compress(P,10),compress(Q,10))
                #self.evaluate=overlap(P,Q)
            if method=='EMD':
                #self.evaluate=getEMD(compress(P),compress(Q))
                self.evaluate=getEMD(P,Q)

        else:
            self.evaluate=0
        return self.evaluate
        #return np.sum(np.array([y*a.f(x)*0.01465 for x,y in zip(angle_,I_)]))
        
def compare(atoms,data,lamb,theta):
    [angles_,Is_]=data
    return XrdStructure(atoms,lamb,theta=theta).evaluate

if __name__ == '__main__':
    lamb=0.6199
    
    import os
    import os.path
    rootdir = os.getcwd()                               
    
    for parent,dirnames,filenames in os.walk(rootdir):    
        for filename in filenames:
            if '123.cif' in filename:
                a=XrdStructure(ase.io.read('POS/'+filename),lamb,filename,[2,7.5])
                
                #for peak in a.peaks:
                #    print('hkl=',peak.hkl,'  theta=',peak.theta/np.pi*360,' I=',peak.I,' multi=',peak.multi,' d=',peak.d)
                if a.evaluate and a.evaluate<350:
                    a.xiajibahua()
                    print(filename,':',a.evaluate)
