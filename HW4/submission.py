"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import matplotlib.pyplot as plt
import helper
# import sympy
import scipy
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import os.path
import findM2
import variables as var

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
        # Replace pass by your implementation
    p1 = pts1*1.0/M
    p2 = pts2*1.0/M
    A = np.vstack([p1[:, 0]*p2[:, 0], p1[:, 0]*p2[:, 1], p1[:, 0],
                   p1[:, 1]*p2[:, 0], p1[:, 1]*p2[:, 1], p1[:, 1],
                   p2[:, 0], p2[:, 1], np.ones(p1.shape[0])])

    U, S, V = np.linalg.svd(A.T)
    F = V[-1, :].reshape(3, 3)
    F = helper._singularize(F)    
    F = helper.refineF(F, p1, p2) 
    T = np.diag([1./M, 1./M, 1])
    F = T.T @ F @ T    
    # print(F)             
    if(os.path.isfile('q2_1.npz')==False):
        np.savez('q2_1.npz',F = F, M = M)
    return F

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1 
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    N, temp = pts1.shape
    W = np.zeros((N,3))
    W_homo = np.zeros((N,4))
    for i in range(N):
        A1 = pts1[i,0]*C1[2,:] - C1[0,:]
        A2 = pts1[i,1]*C1[2,:] - C1[1,:]
        A3 = pts2[i,0]*C2[2,:] - C2[0,:]
        A4 = pts2[i,1]*C2[2,:] - C2[1,:]
        A = np.vstack((A1,A2,A3,A4))
        U, S, V = np.linalg.svd(A)
        p = V[-1, :]
        p = p/p[3]
        W[i, :] = p[0:3]
        W_homo[i, :] = p
    p1_proj = np.matmul(C1,W_homo.T)
    lam1 = p1_proj[-1,:]
    p1_proj = p1_proj/lam1
    p2_proj = np.matmul(C2,W_homo.T)
    lam2 = p2_proj[-1,:]
    p2_proj = p2_proj/lam2
    err1 = np.sum((p1_proj[[0,1],:].T-pts1)**2)
    err2 = np.sum((p2_proj[[0,1],:].T-pts2)**2)
    err = err1 + err2
    print('triangulate err:',err)
    return W,err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pt1 = np.array([[x1],[y1],[1]])
    data = np.load('../data/some_corresp.npz')
    epline = F.dot(pt1)
    H,W,channel = im1.shape
    line_y = np.arange(y1-30,y1+30)
    line_x = (-(epline[1]*line_y+epline[2])/epline[0])
    win = 5
    im11 = ndimage.gaussian_filter(im1, sigma=1, output=np.float64)
    im22 = ndimage.gaussian_filter(im2, sigma=1, output=np.float64)
    min_err = np.inf
    res = 0
    for i in range(60):
        x2 = int(line_x[i])
        y2 = line_y[i]
        if (x2>=win  and x2<= W-win-1 and y2>=win and y2<= H-win-1):
            patch1 = im11[y1-win:y1+win+1, x1-win:x1+win+1,:]
            patch2 = im22[y2-win:y2+win+1, x2-win:x2+win+1,:]
            difference = (patch1-patch2).flatten()
            err= (np.sum(difference**2))
            if (err<min_err):
                min_err = err
                res = i
    if (os.path.isfile('q4_1.npz') == False):
        np.savez('q4_1.npz', F=F, pts1=data['pts1'], pts2=data['pts2'])
    return line_x[res],line_y[res]

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters, tol):
   
    # Replace pass by your implementation
    max_inliers=-1
    F=np.empty([3,3])
    r1=np.empty([8,2])
    r2=np.empty([8,2])
    pts1_homo=np.vstack((pts1.T,np.ones([1,pts1.shape[0]])))
    pts2_homo=np.vstack((pts2.T,np.ones([1,pts1.shape[0]])))
    
    for i in range(nIters):
        print(i)
        total_inliers=0
        idx_rand=np.random.choice(pts1.shape[0],8)
        r1=pts1[idx_rand,:]
        r2=pts2[idx_rand,:] 
        F=eightpoint(r1, r2, M)
        predt_x2_homo=np.dot(F,pts1_homo)
        predt_x2=predt_x2_homo/np.sqrt(np.sum(predt_x2_homo[:2,:]**2,axis=0))        
        err=abs(np.sum(pts2_homo*predt_x2,axis=0))
        inliers_num=err<tol
        total_inliers=inliers_num[inliers_num.T].shape[0]
        print(total_inliers)
        if total_inliers>max_inliers:
            bestF=F
            max_inliers=total_inliers
            inliers=inliers_num
    count=0
    for r in range(len(inliers)):
        if inliers[r]==True:
            count+=1
    Total=len(inliers)
    print(inliers)
    print('count',(count/Total)*100)
            
            
    return bestF,inliers
	

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    if (theta == 0):
        return np.eye(3)
    u = r/theta
    u1 = u[0,0]
    u2 = u[1,0]
    u3 = u[2,0]
    ucrossprod = np.array([[0, -u3, u2],
                       [u3, 0, -u1],
                       [-u2, u1, 0]])
    R = np.eye(3)*np.cos(theta)+(1-np.cos(theta))*u.dot(u.T)+np.sin(theta)*ucrossprod
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T)/2
    xt = np.array([[A[2,1]],[A[0,2]],[A[1,0]]])
    s = np.linalg.norm(xt)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2
    if (s == 0 and c == 1):
        return np.zeros((3,1))
    if (s == 0 and c == -1):
        tw = R + np.eye(3)
        if (np.linalg.norm(tw[:,0]) != 0):
            v = tw[:,0]
        elif(np.linalg.norm(tw[:,1]) != 0):
            v = tw[:,1]
        else:
            v = tw[:,2]
        u = v/np.linalg.norm(v)
        r = u*np.pi
        r1 = r[0,0]
        r2 = r[1,0]
        r3 = r[2,0]
        if ((r1== 0 and r2== 0 and r3<0) or (r1 == 0 and r2<0) or (r1<0)):
            r = -1*r
        return r
    if (s != 0):
        u = xt/s
        theta = np.arctan2(s,c)
        r = theta*u
        return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    C1 = np.matmul(K1, M1)
    num,tw = p1.shape
    P = x[0:3*num].reshape((num,3))
    P_homo = np.hstack((P,np.ones((num,1))))
    r2 = x[3*num:3*num+3].reshape((3,1))
    t2 = x[3*num+3:].reshape((3,1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2,t2))
    C2 = np.matmul(K2,M2)
    p1_proj = np.matmul(C1, P_homo.T)
    p1_proj = p1_proj / p1_proj[-1, :]
    p2_proj = np.matmul(C2, P_homo.T)
    p2_proj = p2_proj / p2_proj[-1, :]
    p1_hat = p1_proj[0:2,:].T
    p2_hat = p2_proj[0:2,:].T
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])]).reshape(4*num,1)

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    residualErr= lambda x: (rodriguesResidual(K1,M1,p1,K2,p2,x).flatten())
    x0 = P_init.flatten()
    R2_init = M2_init[:,0:3]
    r2_init = invRodrigues(R2_init).flatten()
    t2_init = M2_init[:,3].flatten()
    num,tw = p1.shape
    x0 = np.hstack((x0, r2_init,t2_init))
    xF,ier = optimize.leastsq(residualErr, x0)
    print(np.shape(xF))
    print(ier)
    print('proj_error from bundleAdjustment',np.sum(residualErr(xF)**2))
    w = xF[0:3 * num].reshape((num, 3))
    r2 = xF[3*num:3*num+3].reshape((3, 1))
    t2 = xF[3*num+3:].reshape((3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    return M2, w
'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation

    pts1=pts1[pts1[:,2] > Thres]
    p1=pts1[:,:2]
    pts2=pts2[pts2[:,2] > Thres]
    p2=pts2[:,:2]
    pts3=pts3[pts3[:,2] > Thres]
    p3=pts3[:,:2]
    n, temp = p1.shape
    P = np.zeros((n,3))
    Phomo = np.zeros((n,4))
    for i in range(n):
        x1 = p1[i,0]
        y1 = p1[i,1]
        x2 = p2[i,0]
        y2 = p2[i,1]
        x3 = p3[i,0]
        y3 = p3[i,1]
        A1 = x1*C1[2,:] - C1[0,:]
        A2 = y1*C1[2,:] - C1[1,:]
        A3 = x2*C2[2,:] - C2[0,:]
        A4 = y2*C2[2,:] - C2[1,:]
        A5=  x3*C3[2,:] - C3[0,:]
        A6 = y3*C3[2,:] - C3[1,:]
        A = np.vstack((A1,A2,A3,A4,A5,A6))
        print(A.shape)
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        p = p/p[3]
        P[i, :] = p[0:3]
        Phomo[i, :] = p
        # print(p)
    p1_proj = np.matmul(C1,Phomo.T)
    lam1 = p1_proj[-1,:]
    p1_proj = p1_proj/lam1
    p2_proj = np.matmul(C2,Phomo.T)
    lam2 = p2_proj[-1,:]
    p2_proj = p2_proj/lam2
    err1 = np.sum((p1_proj[[0,1],:].T-p1)**2)
    err2 = np.sum((p2_proj[[0,1],:].T-p2)**2)
    err = err1 + err2
    print(err)
    if(os.path.isfile('q6_1.npz')==False):
        np.savez('q6_1.npz',P=P)
    return P,err


