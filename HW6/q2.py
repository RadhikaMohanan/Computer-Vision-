# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    U,S,Vt = np.linalg.svd(I, full_matrices=False)
    S[3:] = 0
    S31 = np.diag(S[:3])
    VT31 = Vt[:3,:]
    B = np.dot(np.sqrt(S31),VT31)
    L = np.dot(U[:,:3],np.sqrt(S31)).T
    return B, L



if __name__ == "__main__":

    # Put your main code here
    
    #Qb
    # I, L0, s = loadData()
    # print(L0)
    # B, L = estimatePseudonormalsUncalibrated(I)
    # print(L)
    # albedos, normals = estimateAlbedosNormals(B)
    # albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    #Qd
    # I, L0, s = loadData()
    # B, L = estimatePseudonormalsUncalibrated(I)
    # albedos, normals = estimateAlbedosNormals(B)
    # surface = estimateShape(normals, s)    
    # min_v, max_v = np.min(surface), np.max(surface)
    # surface = (surface - min_v) / (max_v - min_v)   
    # surface = (surface * 255.).astype('uint8')
    # plotSurface(surface)
    
    #Qe
    # I, L0, s = loadData()
    # B, L = estimatePseudonormalsUncalibrated(I)
    # Nt = enforceIntegrability(B, s)
    # albedos, normals = estimateAlbedosNormals(Nt)
    # surface = estimateShape(normals, s)    
    # min_v, max_v = np.min(surface), np.max(surface)
    # surface = (surface - min_v) / (max_v - min_v)  
    # surface = (surface * 255.).astype('uint8')
    # plotSurface(surface)
    
    #Qf
    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    mu= 10#0.1
    v=0
    lambd=0.1
    G=np.array([[1,0,0],[0,1,0],[mu,v,lambd]])
    print(G)
    G_invT=np.linalg.inv(G).T
    Nt = enforceIntegrability(B, s)
    GTB=np.dot(G_invT,Nt)
    albedos, normals = estimateAlbedosNormals(GTB)
    surface = estimateShape(normals, s)
    
    min_v, max_v = np.min(surface), np.max(surface)
    surface = (surface - min_v) / (max_v - min_v)
    
    surface = (surface * 255.).astype('uint8')
    plotSurface(surface)
    
    
    
  

   
