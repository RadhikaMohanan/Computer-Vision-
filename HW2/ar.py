import numpy as np
import cv2
from loadVid import loadVid
from matchPics import matchPics
from PIL import Image
import imageio

#Import necessary functions
from opts import get_opts
from planarH import computeH
from planarH import computeH_ransac
from planarH import compositeH
#Write script for Q3.1

def func_result(cv_cover, frame, ar_f, opts):
    matches, locs1, locs2 = matchPics(cv_cover, frame, opts)
    x1 = locs1[matches[:,0], 0:2]
    x2 = locs2[matches[:,1], 0:2]
    
    H2to1, inliers = computeH_ransac(x1, x2, opts)
    ar_f = ar_f[45:310,:,:]
    cover_width = cv_cover.shape[1]
    width = int(ar_f.shape[1]/ar_f.shape[0]) * cv_cover.shape[0]

    resized_ar = cv2.resize(ar_f, (width,cv_cover.shape[0]), interpolation = cv2.INTER_AREA)
    h, w, d = resized_ar.shape
    cropped_ar= resized_ar[:,int(w/2)-int(cover_width/2):int(w/2)+int(cover_width/2),:]
    
    result = compositeH(H2to1, cropped_ar, frame)
    
    return result

opts = get_opts()
book = loadVid('../data/book.mov')
src = loadVid('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

a,b,_=book[1].shape
out=cv2.VideoWriter('ar.avi',cv2.VideoWriter_fourcc('X','V','I','D'),2,(b,a))


for i in range(100):
    frame = book[i]
    ar_f = src[i]
    print(i)
    lol = func_result(cv_cover, frame, ar_f, opts)
    out.write(lol)

cv2.destroyAllWindows()
out.release()