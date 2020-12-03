import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from PIL import Image
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
import matplotlib.pyplot as plt


#Write script for Q2.2.4
opts = get_opts()
cv_cover =cv2.imread('../data/cv_cover.jpg')
cv_desk=cv2.imread('../data/cv_desk.png')
hp_cover=cv2.imread('../data/hp_cover.jpg')

# matches,locs1,locs2=matchPics(cv_desk,cv_cover,opts)
# # print(matches.shape)
# x1=locs1[matches[:,0],0:2]
# x2=locs2[matches[:,1],0:2]

# H_ransac, inliers=computeH_ransac(x1,x2,opts)


# # hp_cover=cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))

# hp_new=cv2.warpPerspective(hp_cover, H_ransac, (cv_desk.shape[1],cv_desk.shape[0]))

# # result=compositeH(H_ransac,hp_new,cv_desk)
# cv2.imwrite('warped_image.jpeg',hp_new)
hp_cover_resize = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))
matches,locs1,locs2=matchPics(cv_cover,cv_desk,opts)
# # print(matches.shape)
x1=locs1[matches[:,0],0:2]
x2=locs2[matches[:,1],0:2]
# print(x1,x2)

bestH2to1, inliers=computeH_ransac(x1,x2,opts)

print(np.sum(inliers))
# h, w, _ = cv_desk.shape
# # hp_new=cv2.warpPerspective(hp_cover,bestH2to1, (cv_desk.shape[0],cv_desk.shape[1]))
# hp_new=cv2.warpPerspective(hp_cover, bestH2to1, (w,h))

# print(hp_new)

composite_img = compositeH(bestH2to1, hp_cover_resize , cv_desk)
cv2.imwrite('warped_image_a4.jpeg',composite_img)

plt.imshow(composite_img)