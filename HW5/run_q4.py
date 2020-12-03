import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle
import string
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
gt = [list('TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP'),
                 list('ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'),
                 list('HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR'),
                 list('DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING')]

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'crops')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
   
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
    print('\nDetecting >>>', img)
    
    bboxes, bw = findLetters(im1)
    bboxes.sort(key=lambda x: (x[0]))

    lines, words = [], []
    last_line = 1e8
    line_det_thres = 100
    for bbox in bboxes:
        if abs(last_line - bbox[0]) > line_det_thres:
            if words: lines.append(words)
            words = []
        words.append(bbox)
        last_line = bbox[0]
    if words: lines.append(words)

    counts = []
    bboxes_sorted = []
    for line in lines:
        line.sort(key=lambda x: (x[1]))
        counts.append(len(line))
        bboxes_sorted += line
    bboxes=bboxes_sorted
    counts=np.cumsum(counts)

    plt.imshow(bw, cmap='gray')

    # Preprocessing
    letters_to_detect = []
    for i, bbox in enumerate(bboxes):
        mnr, mnc, mxr, mxc = bbox
        crop = bw[mnr:mxr, mnc:mxc]
        crop = skimage.transform.rescale(crop, 26/max(mxc-mnc, mxr-mnr))
        
        yp, xp = (32 - crop.shape[0])//2, (32 - crop.shape[1])//2
        yo, xo = int(crop.shape[0]%2!=0), int(crop.shape[1]%2!=0)
        letter = np.pad(crop, ((yp, yp + 1*yo), (xp, xp + 1*xo)), 'constant', constant_values=(1, 1))
        skimage.io.imsave('crops/%s_%02d.png'%(img.split('.')[0], i), letter)
        letters_to_detect.append(letter.T.flatten())

        rect = matplotlib.patches.Rectangle((mnc, mnr), mxc-mnc, mxr-mnr, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.show()

    letters_to_detect = np.vstack(letters_to_detect)
    
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    
    hyp1 = forward(letters_to_detect, params, name='layer1', activation=sigmoid)
    probs = forward(hyp1, params, name='output', activation=softmax)
    detects = np.argmax(probs, axis=1)
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    lines = np.split(letters[detects], counts[:-1])
    lines = [ ''.join(line.tolist()) for line in lines ]

    correct = (gt[int(img[1])-1] == letters[detects])
    print('Accuracy: %.2f%%\n'%(correct.sum()*100.0/correct.size))
    for line in lines: print(line)
    