import numpy as np
import skimage.io
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# takes a color image
# returns a list of bounding boxes and black_and_white image
erosion_labels_itrs = 4
bbox_min_area = 1000
erosion_joint_itrs = 1
sigma_blur = 2
closing_itrs_bw = 3
min_aspect = 1.15

def labelSearch(image, morph_iters=erosion_labels_itrs):
    for _ in range(morph_iters): image = skimage.morphology.binary_erosion(image)
    image = skimage.img_as_float(image)
    labels = skimage.measure.label(image-1)
    labels = skimage.segmentation.clear_border(labels)
    regions = skimage.measure.regionprops(labels)
    return labels, regions

def elimJoint(image, bboxes):
    i = 0
    while i < len(bboxes):
        minr, minc, maxr, maxc = bboxes[i]
        if (maxc - minc) / (maxr - minr) > min_aspect:
            letter = np.ones_like(image)
            letter[minr:maxr, minc:maxc] = image[minr:maxr, minc:maxc]
            _, regions = labelSearch(letter, morph_iters=erosion_joint_itrs)
            n = len(regions)
            if n > 1:
                print('Found %d joined letters'%n)
                bboxes = bboxes[:i] + [ region.bbox for region in regions ] + bboxes[i+1:]
                i += (n - 1)
        i += 1

    return bboxes

    
def findLetters(image):
    image = skimage.img_as_float(image)
    image = skimage.color.rgb2gray(image)
    image = image / np.max(image)
    image = skimage.filters.gaussian(image, sigma=sigma_blur)
    threshold = skimage.filters.threshold_otsu(image)
    bw = image > threshold

    iters = (image.shape[0] * image.shape[1])//int(1.5e6) + 2*int(image.shape[1]>4000)
    for _ in range(iters): bw = skimage.morphology.binary_erosion(bw)
    for _ in range(closing_itrs_bw): bw = skimage.morphology.binary_closing(bw)
    bw = skimage.img_as_float(bw)

    labels, regions = labelSearch(bw)
    bboxes = [ region.bbox for region in regions if region.area > bbox_min_area]
    bboxes = elimJoint(bw, bboxes)
    if __name__ == '__main__':
        fig, ax = plt.subplots()
        ax.imshow(bw, cmap='gray')
    
        for bbox in bboxes:
            mnr, mnc, mxr, mxc = bbox
            rect = mpatches.Rectangle((mnc, mnr), mxc-mnc, mxr-mnr, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
    
        plt.show()
    return bboxes, bw

if __name__ == '__main__':
    for i in range(1,5):
        if i==1: img = skimage.io.imread('../images/01_list.jpg')
        if i==2: img = skimage.io.imread('../images/02_letters.jpg')
        if i==3: img = skimage.io.imread('../images/03_haiku.jpg')
        if i==4: img = skimage.io.imread('../images/04_deep.jpg')
        findLetters(img)
           