import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import matlib as mb
def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0
    
    h0,w0 = np.shape(It)
    h1,w1 = np.shape(It1)


    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    
    width = int(x2-x1)
    height = int(y2-y1)
    strt0=np.linspace(0,h0,num=h0,endpoint=False)
    stop0=np.linspace(0,w0,num=w0,endpoint=False)
    strt1=np.linspace(0,h1,num=h1,endpoint=False)
    stop1=np.linspace(0,w1,num=w1,endpoint=False)
    s0 = RectBivariateSpline(strt0,stop0,It)
    s1 = RectBivariateSpline(strt1,stop1,It1)
    
    change = 1
    counter = 1
    x,y = np.mgrid[x1:x2+1:width*1j,y1:y2+1:height*1j]
    while (change > threshold and counter < num_iters):

        dxp = s1.ev(y+p[1], x+p[0],dy = 1).flatten()
        dyp = s1.ev(y+p[1], x+p[0],dx = 1).flatten()
        It1p = s1.ev(y+p[1], x+p[0]).flatten()
        Itp = s0.ev(y, x).flatten()
        A = np.zeros((width*height,2*width*height))
        for i in range(width*height):
            A[i,2*i] = dxp[i]
            A[i,2*i+1] = dyp[i]
        Rs=mb.repmat(np.eye(2),width*height,1)
        A = np.matmul(A,Rs)
        b = np.reshape(Itp - It1p,(width*height,1))
        deltap = np.linalg.pinv(A).dot(b)
        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
    return p
