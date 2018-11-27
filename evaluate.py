import numpy as np
import scipy.optimize

def positivity(f):
    return f 

def fromSrc(f, wp, i, shape):
    fr = np.reshape(f, shape)
    f_sumColi = np.sum(fr[i,:])
    return wp[i] - f_sumColi

def toTgt(f, wq, j, shape):
    fr = np.reshape(f, shape)
    f_sumRowj = np.sum(fr[:,j])
    return wq[j] - f_sumRowj

def maximiseTotalFlow(f, wp, wq): 
    return f.sum() - np.minimum(wp.sum(), wq.sum())

def flow(f, D):
    f = np.reshape(f, D.shape)
    return (f * D).sum()
def getDistance(x,y):
    return [1000,(x-y)**2][abs(x-y)<0.5]

def getDistMatrix(s1, s2):
    numFeats1 = s1.shape[0]
    numFeats2 = s2.shape[0]
    distMatrix = np.zeros((numFeats1, numFeats2))

    for i in range(0, numFeats1):
        for j in range(0, numFeats2):
            distMatrix[i,j] = getDistance(s1[i],s2[j])

    #import scipy.spatial
    #distMatrix = scipy.spatial.distance.cdist(s1, s2)

    return distMatrix

def getFlowMatrix(P, Q, D):
    numFeats1 = P[0].shape[0]
    numFeats2 = Q[0].shape[0]
    shape = (numFeats1, numFeats2) 
    cons1 = [{'type':'ineq', 'fun' : positivity}]
    cons2 = [{'type':'ineq', 'fun' : fromSrc, 'args': (P[1], i, shape,)} for i in range(numFeats1)]
    cons3 = [{'type':'ineq', 'fun' : toTgt, 'args': (Q[1], j, shape,)} for j in range(numFeats2)]
    cons4 = [{'type':'eq', 'fun' : maximiseTotalFlow, 'args': (P[1], Q[1],)}]    
    cons = cons1 + cons2 + cons3 + cons4

    F_guess = np.zeros(D.shape)
    F = scipy.optimize.minimize(flow, F_guess, args=(D,), constraints=cons)
    F = np.reshape(F.x, (numFeats1,numFeats2))
    return F
 
def getEMD(P,Q):
    D = getDistMatrix(P[0], Q[0])
    F = getFlowMatrix(P, Q, D)
    return (F * D).sum() / F.sum()

def overlap(P,Q):
    return np.sum(np.array([x*y for x,y in zip(P[1],Q[1])]))

def comparepeak(P,Q):
    return abs(Q[0][np.argmax(Q[1])]-P[0][np.argmax(P[1])])<0.5
        