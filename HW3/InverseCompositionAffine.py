import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
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
    x, y = np.mgrid[0:w0, 0:h0]
    x = np.reshape(x, (1, height * width))
    y = np.reshape(y, (1, height * width))
    hom_cor = np.vstack((x, y, np.ones((1, height * width))))
    dxT = s0.ev(y, x, dy=1).flatten()
    dyT = s0.ev(y, x, dx=1).flatten()
    Itp = s0.ev(y, x).flatten()
    x = np.reshape(x, (height * width, 1))
    y = np.reshape(y, (height * width, 1))
    dxT = np.reshape(dxT, (height * width, 1))
    dyT = np.reshape(dyT, (height * width, 1))
    A1 = np.multiply(x, dxT)
    A2 = np.multiply(y, dxT)
    A11 = np.multiply(x, dyT)
    A22 = np.multiply(y, dyT)
    Ap = np.hstack((A1, A2, dxT, A11, A22, dyT))
    
    precompute = np.matmul(np.linalg.pinv(np.matmul(Ap.T,Ap)),Ap.T)
    while (change > threshold and counter < num_iters):
        M = np.array([[1 + p[0], p[1], p[2]],
                      [p[3], 1 + p[4], p[5]],
                      [0, 0, 1]])
        coorp = np.matmul(M, hom_cor)
        xp = coorp[0]
        yp = coorp[1]

        It1p = s1.ev(yp, xp).flatten()

        b = np.reshape(Itp - It1p, (len(xp), 1))
        deltap = precompute.dot(b)

        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
        counter += 1
    M = np.array([[1 + p[0], p[1], p[2]],
                  [p[3], 1 + p[4], p[5]],
                  [0, 0, 1]])
    return M
