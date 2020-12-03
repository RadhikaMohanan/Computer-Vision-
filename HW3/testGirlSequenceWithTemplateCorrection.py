import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect0 = [280, 152, 330, 318]
rect = rect0[:]
rects = rect[:]
h,w,frames = np.shape(seq)
update = True
It = seq[:,:,0]
p0 = np.zeros(2)
for frame in range(frames-1):
    It1 = seq[:,:,frame+1]
    p = LucasKanade.LucasKanade(It, It1, rect, threshold, num_iters,p0)
    pdp= p + [rect[0] - rect0[0],rect[1]-rect0[1]]
    p_star = LucasKanade.LucasKanade(seq[:,:,0],It1,rect0,threshold, num_iters,pdp)
    change = np.linalg.norm(pdp-p_star)
    if change<threshold:
        p_2= (p_star - [rect[0] - rect0[0],rect[1]-rect0[1]])
        rect[0] += p_2[0]
        rect[2] += p_2[0]
        rect[1] += p_2[1]
        rect[3] += p_2[1]
        It = seq[:, :, frame+1]
        rects = np.vstack((rects, rect))
        p0 = np.zeros(2)
    else:

        rects = np.vstack((rects,[rect[0]+p[0],rect[1]+p[1],rect[2]+p[0],rect[3]+p[1]]))
        p0 = p

np.save('girlseqrects-wcrt.npy', rects)
girlseqrects = np.load('girlseqrects.npy')
girlseqrects_ct = np.load('girlseqrects-wcrt.npy')
frame_req = [1,20,40,60,80] 

for ind in range(len(frame_req)):
    i = frame_req[ind]
    fig = plt.figure()
    frame =seq[:,:,i]
    rect_nc= girlseqrects[i,:]
    rect_ct = girlseqrects_ct[i,:]
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    patch1 = patches.Rectangle((rect_nc[0],rect_nc[1]),(rect_nc[2]-rect_nc[0]),(rect_nc[3]-rect_nc[1]),edgecolor = 'b',facecolor='none',linewidth=2)
    patch2 = patches.Rectangle((rect_ct[0],rect_ct[1]),(rect_ct[2]-rect_ct[0]),(rect_ct[3]-rect_ct[1]),edgecolor = 'r',facecolor='none',linewidth=2)

    ax = plt.gca()
    ax.add_patch(patch1)
    ax.add_patch(patch2)
    fig.savefig('girlseq-wcrtframe' + str(i) + '.png',bbox_inches='tight')