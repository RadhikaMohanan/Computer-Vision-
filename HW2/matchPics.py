import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from opts import get_opts
from skimage.color import rgb2gray

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

	#Convert Images to GrayScale
    # Img1 = rgb2gray(I1)
    # Img2 = rgb2gray(I2)
    Img1=np.dot(I1,[0.299,0.587,0.114])
    Img2=np.dot(I2,[0.299,0.587,0.114])
    # Img1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    # Img2= cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
    locs11=corner_detection(Img1, sigma)
    locs21=corner_detection(Img2, sigma)
	
	
	#Obtain descriptors for the computed feature locations
    desc1,locs1=computeBrief(Img1,locs11)
    desc2,locs2=computeBrief(Img2,locs21)
	

	#Match features using the descriptors
    matches=briefMatch(desc1,desc2,ratio)
    return matches, locs1, locs2
