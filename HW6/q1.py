# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from utils import integrateFrankot
from skimage.color import rgb2xyz
import cv2
from matplotlib import cm
def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    X = np.linspace(0, res[0]-1, int(res[0]))
    Y = np.linspace(0, res[1]-1, int(res[1]))
    [x,y] = np.meshgrid(X, Y)
    
    rad1 = int(rad // pxSize)
    res1 = rad1**2 - ((res[0]//2 + int(center[0]//pxSize) - x)**2 + (res[1]//2 - int(center[1]//pxSize) - y)**2)
    mask = (res1 >= 0)
    res1 = res1 * mask   
    idx = np.where(res1 == 0)
    res1[idx[0],idx[1]] = 1   
    p1 = (x-res[0]//2) / np.sqrt(res1)
    q1 = (y-res[1]//2) / np.sqrt(res1)
    
    albedo = 0.5
    R1 = (albedo * (light[0] * p1 - light[1] * q1 + light[2])) / np.sqrt(1 + p1**2 + q1**2)
    R1 = R1 * mask
    idx = np.where(R1 < 0)
    R1[idx[0], idx[1]] = 0   
    image = R1 / np.max(R1) * 255
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    A=[]
    for i in range(1,8):
        im = cv2.imread(path+"input_"+str(i)+".tif", -1) 
        I_xyz =rgb2xyz(im)
        I_y=I_xyz[:,:,1].reshape(I_xyz.shape[0]*I_xyz.shape[1],1)
        A.append(I_y)
    A=np.asarray(A)
    I=A.reshape(A.shape[0],A.shape[1])
    L = np.load(path+'sources.npy').T
    s = im.shape[:2]
    return I, L, s

def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.lstsq(L.T, I, rcond=None)[0]
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = []
    
    for i in range(B.shape[1]):
        magnitude = np.linalg.norm(B[:,i])
        albedos.append(magnitude)
        B[:,i] /= magnitude
    albedos=np.asarray(albedos)
    normals = B
    
    print(normals.shape)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = np.reshape(albedos, s)
    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    normals += abs(np.min(normals))
    normals /= np.max(normals)
    l1 = normals[0,:].reshape(s)
    l2 = normals[1,:].reshape(s)
    l3 = normals[2,:].reshape(s)
    normalIm = np.dstack((np.dstack((l1, l2)), l3))
    plt.imshow(normalIm, cmap='rainbow')
    plt.show()
    return albedoIm, normalIm 


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    zx = np.reshape(normals[0,:] / (-normals[2,:]), s)
    zy = np.reshape(normals[1,:] / (-normals[2,:]), s)
    surface = integrateFrankot(zx, zy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    h1, w1 = surface.shape
    y1, x1 = range(h1), range(w1)
    X, Y = np.meshgrid(x1, y1)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, surface, edgecolor='none', cmap=cm.coolwarm)
    ax.set_title('Surface plot')
    plt.show()

def initials(I):
    u, sing, vh = np.linalg.svd(I,full_matrices=False)
    return sing


if __name__ == '__main__':

    # Put your main code here
    #Qb
    # l1 = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    # l2 = np.array([1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])
    # l3 = np.array([-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)])
    # light = np.vstack((l1,l2,l3))
    # center = np.array([0,0,0])
    # radius = 0.75
    # frame = np.array([3840, 2160])
    # pxSize=0.0007
    # for i in range(0,3):
    #   img = renderNDotLSphere(center, radius, light[i,:], pxSize, frame)

    #   plt.imshow(img, cmap='gray')
    #   plt.show()
    
    # Qc 
    I,L,s=loadData(path = "../data/")
    
    # Qd
    sing= initials(I)
    print(sing)
   
    #Qe
    B = estimatePseudonormalsCalibrated(I,L)
    albedos, normals=estimateAlbedosNormals(B)
    
    #Qf
    # albedoIm, normalIm =displayAlbedosNormals(albedos, normals, s)
    
    #Qi
    surface=estimateShape(normals, s)
    minv, maxv = np.min(surface), np.max(surface)
    surface = (surface - minv) / (maxv - minv)    
    surface = (surface * 255.).astype('uint8')
    plotSurface(surface)
    
  

    