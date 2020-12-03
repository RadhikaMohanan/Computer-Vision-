import os, math, multiprocessing
from os.path import join
from copy import copy
from opts import get_opts
import numpy as np
from PIL import Image
from image_split import split
import visual_words
from PIL import Image
import scipy.ndimage
import skimage.color
import imageio
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.spatial.distance
import util
import numpy as np
from sklearn.metrics import confusion_matrix

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape
    '''
    opts = get_opts()
    K = opts.K
    # ----- TODO -----
    H,W= np.shape(wordmap)
    data=np.reshape(wordmap,(H*W,1))
    hist_bins=np.linspace(0, K, num=K+1, endpoint=True)                  
    hist1, bin_edge = np.histogram(data, bins=hist_bins)
    hist=hist1/np.linalg.norm(hist1,ord=1)
    hist = np.reshape(hist,(1,K))
    #print('complete5')
    return (hist)


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    H,W= np.shape(wordmap)
    opts = get_opts()    
    K = opts.K
    L = opts.L

    weights = []
    for l in range(L):
        if l == 0 or l == 1:
            weights.append(2**(-(L-1)))
        else:
            weights.append(2**(l-L))
    hist_all= np.array([], dtype=np.int64).reshape(1,0)
    for i in range(len(weights)):
        layer = len(weights)-1-i
        weight = weights[len(weights)-1-i]
        sec_H = int(H/(2**layer))
        sec_W = int(W/(2**layer))
        for row in range(2**layer):
            for col in range(2**layer):
                subword = wordmap[sec_H*row:sec_H*(row+1),sec_W*col:sec_W*(col+1)]
                hist = get_feature_from_wordmap(opts,subword)
                hist_all = np.hstack([ hist_all,hist*weight])
    #print('complete6')
    return hist_all

    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    opts = get_opts()
    img=imageio.imread('../data/'+img_path)
    img=img.astype('float')/255

    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    #print('complete7')
    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    opts = get_opts()
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    train_data = np.asarray(train_files)
    K = dictionary.shape[0]
    n_train = train_data.shape[0]
    labels = np.asarray(train_labels)
    M = int(K*(4**SPM_layer_num-1)/3)
    features=np.empty(M).reshape((1,M))
    for i in range(n_train):
        img_path = train_data[i]
        feat = get_image_feature(opts,img_path,dictionary) 
        features= np.vstack([features, feat]) 
    np.save('trained_system.npy', features)
    np.save('labels.npy', labels)
    np.savez('trained_system.npz',features = features,labels = labels,dictionary = dictionary,SPM_layer_num = SPM_layer_num)
    #print('complete8')


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    intersection = np.minimum(word_hist,histograms)
    similarity = np.sum(intersection,axis = 1)
    #print('complete9')
    return similarity
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    opts = get_opts()
    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']

    train_labels = trained_system['labels']

    
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    test_data  = np.asarray(test_files)
    test_data=test_data.astype(str)
    test_labels = np.asarray(test_labels)
    n_test = len(test_files)

    
    predict_labels=list()
    ii=0
    for t in range(n_test):
        img_path =  test_data[t]
        test_feature = get_image_feature(opts,img_path,dictionary)
        similarity = distance_to_set(test_feature,features)
        idx = np.argmax(similarity)
        print(idx)
        ii+=1
        print('count of idx',ii)
        pl = train_labels[idx]
        predict_labels.append(pl)
    confusion = np.zeros((8,8))
    count=0
    for i in range(n_test):
        x=predict_labels[i]
        y=test_labels[i]
        confusion[x,y]+=1
        if predict_labels[i]==test_labels[i]:
            count+=1
    accuracy=count/len(test_labels)

    return(confusion,accuracy)
        
    #print('complete10')

