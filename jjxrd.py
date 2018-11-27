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

def compress(P,n=20,l=None):
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
        
class XrdStructure():
    def __init__(self,atoms,lamb,data,name='QAQ',theta=None):
        self.atoms=atoms
        self.lattice=self.atoms.get_cell_lengths_and_angles()
        self.reciprocal_lattice=self.atoms.get_reciprocal_cell()*2*np.pi
        self.lamb=lamb
        [self.angles_,self.Is_]=data
        self.name=name
        if theta is None:
            [self.thetamin,self.thetamax]=[np.min(self.angles_-0.5)/2,np.max(self.angles_+0.5)/2]
        else:
            [self.thetamin,self.thetamax]=theta
        self.Khmax=4*np.pi*np.sin(self.thetamax/180*np.pi)/lamb
        self.Khmin=4*np.pi*np.sin(self.thetamin/180*np.pi)/lamb
        self.peaks=[]
        
        self.getallhkl()
        self.angles=np.array([peak.theta/np.pi*360 for peak in self.peaks])
        self.Is=np.array([peak.get_I() for peak in self.peaks])
        self.Is=self.Is/np.sum(self.Is)
        self.evaluate()
    
    def xiajibahua(self):
        plt.figure()
        angle=np.arange(2*self.thetamin,2*self.thetamax,0.01)
        I=np.array([self.f(x,big=False) for x in angle])
        plt.plot(angle,I,label=self.name)
        plt.plot(self.angle_,self.I_,label='shiyan')
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
        P=(self.angles_,self.Is_)
        Q=(self.angles,self.Is)
        if comparepeak(P,Q):
            if method=='overlap':
                P=(self.angle_,self.I_)
                Q=(self.angle_,np.array([self.f(x) for x in self.angle_]))
                self.evaluate=2*overlap(compress(P,20),compress(Q,20))+overlap(compress(P,10),compress(Q,10))
            if method=='EMD':
                self.evaluate=getEMD(P,Q)

        else:
            self.evaluate=0
        return self.evaluate

        
def compareXrd(atoms,data,lamb,theta=None):
    return XrdStructure(atoms,lamb,data,theta=theta).evaluate
