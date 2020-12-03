import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')#1e3
parser.add_argument('--threshold', type=float, default=0.01, help='dp threshold of Lucas-Kanade for terminating optimization')#1e-2
parser.add_argument('--tolerance', type=float, default=0.02, help='binary threshold of intensity difference when computing the mask')#0.4
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

imH,imW,frames = np.shape(seq)

for frame in range(frames-1):
    image1 = seq[:,:,frame]
    image2 = seq[:,:,frame+1]
    mask = SubtractDominantMotion.SubtractDominantMotion(image1,image2, threshold, num_iters, tolerance)
    objects = np.where(mask == 1)
    if ((frame) % 30 == 29):
        pic = plt.figure()
        plt.imshow(image2, cmap='gray')
        plt.axis('off')
        fig,= plt.plot(objects[1],objects[0] ,'*')
        fig.set_markerfacecolor((0, 0, 1, 1))
        pic.savefig('antseqframe'+str(frame+1)+'.png', bbox_inches='tight')
        