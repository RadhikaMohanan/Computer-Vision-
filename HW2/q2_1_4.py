import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
from planarH import computeH
from planarH import computeH_norm
from planarH import computeH_ransac

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')


matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

#display matched features
plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# x1=locs1[matches[:,0],0:2]
# x2=locs2[matches[:,1],0:2]
# x1_centroid=[np.mean(x1[:,0]),np.mean(x1[:,1])]
# x2_centroid=[np.mean(x2[:,0]),np.mean(x2[:,1])]
# p1=np.zeros((x1.shape[0],1), dtype=float)
# p2=np.zeros((x2.shape[0],1), dtype=float)
# for i in range(x1.shape[0]):
#     p1[i]=np.asarray(np.sqrt((x1[i,0]-x1_centroid[0])**2+(x1[i,1]-x1_centroid[1])**2))
#     p2[i]=np.asarray(np.sqrt((x2[i,0]-x2_centroid[0])**2+(x2[i,1]-x2_centroid[1])**2))
# # p1=print('Hcalculated',computeH_norm(x1[:4], x2[:4]))

# # # print('OpencvH',cv2.findHomography(x2[:4, ::-1], x1[:4, ::-1]))
# print(computeH_ransac(locs1=x1, locs2=x2, opts=opts))

