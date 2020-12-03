import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import math
from opts import get_opts
import imageio
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.spatial.distance
import util

# Globals
PROGRESS = 0
PROGRESS_LOCK = multiprocessing.Lock()
NPROC = util.get_num_CPU()

sample_response_path= '../data/sampled_responses'

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    opts = get_opts()
    filter_scales = opts.filter_scales
    # ----- TODO -----
    if len(img.shape) < 3:
        img = np.dstack([img] * 3)

    # Convertion of higher channel images to 3 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    
    # Convert RGB to LAB color space
    img_lab = skimage.color.rgb2lab(img)
    filter_responses = []
    for s in filter_scales:
        # Gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:, :, i], sigma=s,mode='reflect'))
        
        # Laplacian of gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_laplace(img_lab[:, :, i], sigma=s,mode='reflect'))
        
        # X axis derivative of gaussian
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:, :, i], sigma=s, order=[1, 0],mode='reflect'))
        
        # Y axis derivative of gaussian 
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:, :, i], sigma=s, order=[0, 1],mode='reflect'))
    # Stack all 4 filters * 3 channels * 5 sigmas
    filter_responses = np.dstack(filter_responses)
    ##print('complete1')
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    global PROGRESS
    with PROGRESS_LOCK: PROGRESS += NPROC
    opts = get_opts()
    i,alpha,img_path=args

    img=imageio.imread('../data/'+img_path)
    img=img.astype('float')/255

    filter_responses = extract_filter_responses(opts, img)

    #Random sampling of responses:
    mm,nn,kk = np.shape(filter_responses)
    sampled_response = np.reshape(filter_responses,(mm*nn,kk))
    idx = np.random.randint(mm*nn, size= alpha)
    sampled_response = sampled_response[idx,:]
    ##print('complete2')
    return sampled_response
    
    #Saving the sampled responses
    np.save('%s%d'%(sample_response_path, i), np.asarray(sampled_response))


def compute_dictionary(opts, n_worker=8):
    '''
    Creates the dictionary of visual words by clustering using k-means.
1
    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''
    opts = get_opts()
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    train_data=np.asarray(train_files)

    if not os.path.exists(sample_response_path):
        os.makedirs(sample_response_path)
    n_train = train_data.shape[0]
    args = [ (i, alpha, train_data[i]) for i in range(n_train)]

    pool = multiprocessing.Pool(processes=n_worker)
    A= pool.map(compute_dictionary_one_image, args)
    Response=A[0]
    for i in range(1,len(A)):
        Response=np.concatenate((Response,A[i]),axis=0)
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs =n_worker).fit(Response)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary.npy',dictionary)
    ##print('complete3')
    return dictionary

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    opts = get_opts()
    # ----- TODO -----
       # Extract filter responses and reshape to (H*W, 3F)
    filter_responses = extract_filter_responses(opts,img)
    filter_responses = filter_responses.reshape(filter_responses.shape[0] * filter_responses.shape[1], filter_responses.shape[2])

    # Calculate euclidean distances (H*W, K) for each pixel with dictionary 
    euclidean_distances = scipy.spatial.distance.cdist(filter_responses, dictionary)

    # Create a wordmap with every pixel equal to the k index with least distance and reshape to (H, W)
    wordmap = np.asarray([ np.argmin(pixel) for pixel in euclidean_distances ]).reshape(img.shape[0], img.shape[1])

    return wordmap


