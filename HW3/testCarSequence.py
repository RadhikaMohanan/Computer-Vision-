import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]     
h,w,frames = np.shape(seq)
rects = rect[:]
for frame in range(frames-1):
    It = seq[:,:,frame]
    It1 = seq[:,:,frame+1]
    p = LucasKanade.LucasKanade(It,It1,rect, threshold, num_iters)
    rect[0] += p[0]
    rect[2] += p[0]
    rect[1] += p[1]
    rect[3] += p[1]
    rects = np.vstack((rects,rect))
    print
    if ((frame) % 100 == 99 or frame == 0):
        fig = plt.figure()
        plt.imshow(It1,cmap = 'gray')
        plt.axis('off')
        plt.axis('tight')
        patch = patches.Rectangle((rect[0],rect[1]),(rect[2]-rect[0]),(rect[3]-rect[1]),edgecolor = 'r',facecolor='none',linewidth=2)
        ax = plt.gca()
        ax.add_patch(patch)
        fig.savefig('carseqframe'+str(frame+1)+'.png',bbox_inches='tight')
np.save('carseqrects.npy', rects)