'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path
import submission as sub
import helper
import findM2
import variables as var


def q42 ():
    data = np.load('../data/some_corresp.npz')
    Ks = np.load('../data/intrinsics.npz')
    K1 = Ks['K1']
    K2 = Ks['K2']
    pts1 = data['pts1']
    pts2 = data['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(np.shape(im1))
    F = sub.eightpoint(data['pts1'], data['pts2'], M)
    M1, C1, M2, C2, F = findM2.findM2(pts1, pts2, F, K1, K2)
    data = np.load('../data/templeCoords.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    x1 = data['x1']
    y1 = data['y1']
    print(x1.shape)
    n1, t1 = x1.shape
    pts1 = np.hstack((x1, y1))
    pts2 = np.zeros((n1, 2))
    for i in range(n1):
        x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i, 0], y1[i, 0])
        pts2[i, 0] = x2
        pts2[i, 1] = y2
    W, err = sub.triangulate(C1, pts1, C2, pts2)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(W[:, 0], W[:, 1], W[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim3d(3.4, 4.2)
    ax.set_xlim3d(-0.8, 0.6)
    plt.show()
    if (os.path.isfile('q4_2.npz') == False):
        np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)


def q53():
    data = np.load('../data/some_corresp_noisy.npz')

    Ks = np.load('../data/intrinsics.npz')
    K1 = Ks['K1']
    K2 = Ks['K2']
    pts1 = data['pts1']
    pts2 = data['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(np.shape(im1))
    F,inliers = sub.ransacF(data['pts1'], data['pts2'], M,var.nIters,var.tol)
    print(F)
    E = sub.essentialMatrix(F, K1, K2)
    M1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
    M2s = helper.camera2(E)
    row, col, num = np.shape(M2s)
    C1 = np.matmul(K1, M1)
    for i in range(num):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = sub.triangulate(C1, pts1[inliers], C2, pts2[inliers])
        if (np.all(P[:, 2] > 0)):
            break
    M2, P = sub.bundleAdjustment(K1, M1, pts1[inliers], K2, M2, pts2[inliers], P)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def q61():
# data = np.load('../data/some_corresp.npz')
    df1 = np.load('../data/q6/time1.npz')
    image=plt.imread('../data/q6/cam1_time1.jpg')
    Thres=var.Thres
    print(df1.files)
    pts1=df1['pts1']
    
    pts2=df1['pts2']
    
    pts3=df1['pts3']
    
    M1=df1['M1']
    M2=df1['M2']
    M3=df1['M3']
    K1=df1['K1']
    K2=df1['K2']
    K3=df1['K3']
    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)
    C3 = np.matmul(K3, M3)
    
    P,err=sub.MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres)
    helper.plot_3d_keypoint(P)

if __name__ == '__main__':
    q42()
    # q53()
