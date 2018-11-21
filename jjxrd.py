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

def f(x,h,mu):
    sig=0.1
    f=0
    for i in range(len(h)):
        f+=h[i]/sig/np.sqrt(2*np.pi)*np.e**(-0.5*(x-mu[i])**2/sig**2)
    return f

class XRD():
    def __init__(self,hkl,theta,Kh,atoms):
        self.hkl=hkl
        self.multi=1
        self.theta=theta
        self.Kh=Kh
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
    def __init__(self,atoms,lamb):
        self.atoms=atoms
        self.lattice=self.atoms.get_cell_lengths_and_angles()
        self.reciprocal_lattice=self.atoms.get_reciprocal_cell()*2*np.pi
        self.dmin=lamb/2
        self.peaks=[]
    
    def xiajibahua(self):
        angle=np.arange(0,180,0.01)
        I=np.array([f(x,self.Is,self.angles) for x in angle])
        plt.scatter(angle,I,s=3)
        
    def initialize(self):
        self.angles=[]
        self.Is=[]
        self.getallhkl()
        for peak in self.peaks:
            peak.get_I()
            self.angles.append(peak.theta/np.pi*360)
            self.Is.append(peak.I)
        
    def getallhkl(self):
        index_range = np.arange(8, -9,-1)
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
        if 2*np.pi<self.getKh(hkl)*self.dmin or self.getKh(hkl)==0:
            return False
        else:
            return np.arcsin(self.dmin/(2*np.pi/self.getKh(hkl)))

    def getKh(self,hkl):
        Kh=np.dot(hkl,self.reciprocal_lattice)
        return np.sqrt(np.dot(Kh,Kh))

lamb=1.54
atoms=ase.io.read('H.vasp')
a=XrdStructure(atoms,lamb)
a.initialize()
for peak in a.peaks:
    print('hkl=',peak.hkl,'  theta=',peak.theta/np.pi*360,' I=',peak.I,' multi=',peak.multi)
a.xiajibahua()