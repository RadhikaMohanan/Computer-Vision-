import numpy as np
from scipy.interpolate import RectBivariateSpline
def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    """
    p0 = np.zeros(6)
    h0, w0 = np.shape(It)
    h1, w1 = np.shape(It1)
 
    width = w0
    height = h0
    s0 = RectBivariateSpline(np.linspace(0, h0, num=h0, endpoint=False),
                                  np.linspace(0, w0, num=w0, endpoint=False), It)
    s1 = RectBivariateSpline(np.linspace(0, h1, num=h1, endpoint=False),
                                  np.linspace(0, w1, num=w1, endpoint=False), It1)
    p = p0
    change = 1
    counter = 1
    x, y = np.mgrid[0:w0,0:h0]
    x = np.reshape(x,(1,height*width))
    y = np.reshape(y,(1,height*width))
    hom_cor = np.vstack((x, y, np.ones((1, height * width))))

    while (change > threshold and counter < num_iters):
        M = np.array([[1+p[0], p[1],p[2]],
                      [p[3],1+p[4],p[5]],
                      [0,0,1]])
        coorp = np.matmul(M,hom_cor)
        xp = coorp[0]
        yp = coorp[1]
        #Selecting only valid xp and yp
        xout = (np.where(xp>=w0) or np.where(xp < 0))

        yout = (np.where(yp>=h0) or np.where(yp < 0))

        if (np.shape(xout)[1] == 0 and np.shape(yout)[1] == 0):
            bad = []
        elif (np.shape(xout)[1] != 0 and np.shape(yout)[1] ==0):
            bad = xout

        elif (np.shape(xout)[1] == 0 and np.shape(yout)[1] !=0):
            bad = yout
        else:

            bad = np.unique(np.concatenate((xout,yout),0))

        xnew = np.delete(x,bad)
        ynew = np.delete(y,bad)
        xp = np.delete(xp,bad)
        yp = np.delete(yp,bad)
        dxp = s1.ev(yp, xp, dy=1).flatten()
        dyp = s1.ev(yp, xp, dx=1).flatten()
        It1p = s1.ev(yp, xp).flatten()
        Itp = s0.ev(ynew, xnew).flatten()

        xnew =  np.reshape(xnew,(len(xnew),1))
        ynew =  np.reshape(ynew,(len(ynew),1))
        xp = np.reshape(xp,(len(xp),1))
        yp = np.reshape(yp,(len(yp),1))
        dxp = np.reshape(dxp, (len(dxp), 1))
        dyp = np.reshape(dyp, (len(dyp), 1))

        A1 = np.multiply(xnew,dxp)
        A2 = np.multiply(ynew,dxp)
        A11 = np.multiply(xnew,dyp)
        A22 = np.multiply(ynew,dyp)
        A = np.hstack((A1,A2,dxp,A11,A22,dyp))

        b = np.reshape(Itp - It1p, (len(xp), 1))
        deltap = np.linalg.pinv(A).dot(b)

        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
        counter += 1
        
    M = np.array([[1+p[0], p[1],p[2]],
                      [p[3],1+p[4],p[5]],
                      [0,0,1]])
    return M