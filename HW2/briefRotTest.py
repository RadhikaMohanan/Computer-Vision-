import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from helper import plotMatches
from opts import get_opts

#Q2.1.6
#Read the image and convert to grayscale, if necessary
opts=get_opts()
img=cv2.imread('../data/cv_cover.jpg')
hist_match=list()

for i in range(36):
	#Rotate Image
    img_rotate=rotate(img,10*(i+1))
    matches,locs1,locs2=matchPics(img,img_rotate,opts)
	#Compute features, descriptors and Match features
    hist_match.append(len(matches))

	#Update histogram
    if(i%11==0 or i==0):
        plotMatches(img,img_rotate,matches,locs1,locs2)
print(hist_match)



#Display histogram
plt.hist(hist_match)
plt.ylabel("Number of Matches")
