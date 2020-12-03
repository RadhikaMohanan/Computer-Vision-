import numpy as np
import LucasKanadeAffine
import InverseCompositionAffine
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.morphology import binary_opening, binary_closing
from scipy.ndimage import affine_transform

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    

    mask = np.zeros(image1.shape, dtype=bool)
    M = LucasKanadeAffine.LucasKanadeAffine(image1,image2, threshold, num_iters)
    M = np.linalg.inv(M)
    # M = InverseCompositionAffine.InverseCompositionAffine(image1,image2, threshold, num_iters)

    warp_image1 = scipy.ndimage.affine_transform(image1,M[0:2,0:2],offset = M[0:2,2],output_shape = image2.shape)
    diff = abs(warp_image1 - image2)
    mask[diff > tolerance] = 1
    mask[warp_image1==0.] = 0
    
    
    mask = scipy.ndimage.morphology.binary_erosion(mask,structure = np.eye(2),iterations= 1)
    mask = scipy.ndimage.morphology.binary_erosion(mask,structure = np.ones((2,1)),iterations= 1)
    mask = scipy.ndimage.morphology.binary_dilation(mask,iterations = 1)  


    return mask
