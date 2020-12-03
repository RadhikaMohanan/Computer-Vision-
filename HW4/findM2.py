'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission as sub
import helper
import matplotlib.pyplot as plt
import os.path


def findM2 (pts1, pts2, F, K1, K2):
    
    E = sub.essentialMatrix(F, K1, K2)
    # print(E)
    M1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
    M2s = helper.camera2(E)
    row, col, num = np.shape(M2s)
    
    C1 = np.matmul(K1, M1)

    for i in range(num):
        M2 = M2s[:, :, i]
        print(M2)
        C2 = np.matmul(K2, M2)
        P, err = sub.triangulate(C1, pts1, C2, pts2)
        if (np.all(P[:,2] > 0)) :
            err_best = err
            P_best = P
            M2_best = M2
            C2_best = C2
            break
    # print('err_best: ', err_best)
    # print('M2_best: ', M2_best )
    # print('C2_best: ', C2_best  )
    # print('P_best: ', P_best )

    if(os.path.isfile('q3_3.npz')==False):
        np.savez('q3_3.npz',  M2=M2_best, C2=C2_best, P=P_best)
    return M1,C1,M2_best,C2_best,F


if __name__ == '__main__':
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
    findM2(pts1, pts2, F, K1, K2)
