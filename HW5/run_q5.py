import numpy as np
import scipy.io
from nn import *
from collections import Counter
import skimage.measure
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()
examples, dimension = train_x.shape
valid_examples = valid_x.shape[0]


initialize_weights(dimension,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'hidden1')
initialize_weights(hidden_size,hidden_size,params,'hidden2')
initialize_weights(hidden_size,dimension,params,'output')


params['m_Wlayer1'] = np.zeros((dimension,hidden_size))
params['m_Whidden1'] = np.zeros((hidden_size,hidden_size))
params['m_Whidden2'] = np.zeros((hidden_size,hidden_size))
params['m_Woutput'] = np.zeros((hidden_size,dimension))
params['m_blayer1'] = np.zeros(hidden_size)
params['m_bhidden1'] = np.zeros(hidden_size)
params['m_bhidden2'] = np.zeros(hidden_size)
params['m_boutput'] = np.zeros(dimension)

training_loss_data = []
valid_loss_data = []
training_acc_data = []
valid_acc_data = []
# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        hypo1 = forward(xb, params, 'layer1', relu)
        hypo2 = forward(hypo1, params, 'hidden1', relu)
        hypo3 = forward(hypo2, params, 'hidden2', relu)
        out = forward(hypo3, params, 'output', sigmoid)

        loss = np.sum((xb - out)**2)
        total_loss += loss
        delta1 = 2*(out-xb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2,params,'hidden2' , relu_deriv)
        delta4 = backwards(delta3,params,'hidden1' , relu_deriv)
        backwards(delta3,params,'layer1' , relu_deriv)
        
        
        params['m_Wlayer1'] = 0.9* params['m_Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['Wlayer1'] += params['m_Wlayer1']
        params['m_blayer1'] = 0.9 * params['m_blayer1'] - learning_rate * params['grad_blayer1']
        params['blayer1'] += params['m_blayer1']
        params['m_Whidden1'] = 0.9 * params['m_Whidden1'] - learning_rate * params['grad_Whidden1']
        params['Whidden1'] += params['m_Whidden1']
        params['m_bhidden1'] = 0.9 * params['m_bhidden1'] - learning_rate * params['grad_bhidden1']
        params['bhidden1'] += params['m_bhidden1']
        params['m_Whidden2'] = 0.9 * params['m_Whidden2'] - learning_rate * params['grad_Whidden2']
        params['Whidden2'] += params['m_Whidden2']
        params['m_bhidden2'] = 0.9 * params['m_bhidden2'] - learning_rate * params['grad_bhidden2']
        params['bhidden2'] += params['m_bhidden2']
        params['m_Woutput'] = 0.9 * params['m_Woutput'] - learning_rate * params['grad_Woutput']
        params['Woutput'] += params['m_Woutput']
        params['m_boutput'] = 0.9 * params[' '] - learning_rate * params['grad_boutput']
        params['boutput'] += params['m_boutput']
     
    tr_loss = total_loss / examples    
   
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,tr_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
    
    valid_hyp1 = forward(valid_x, params, 'layer1', relu)
    valid_hyp2 = forward(valid_hyp1, params, 'hidden1', relu)
    valid_hyp3 = forward(valid_hyp2, params, 'hidden2', relu)
    valid_out = forward(valid_hyp3, params, 'output', sigmoid)
    valid_loss = np.sum((valid_x - valid_out) ** 2)
    training_loss_data.append(total_loss / examples)  
    valid_loss_data.append(valid_loss / valid_examples)
    
plt.figure(0)
plt.plot(np.arange(max_iters),training_loss_data,'r')
plt.plot(np.arange(max_iters),valid_loss_data,'b')
plt.legend(['training loss','valid loss'])
plt.xlabel('max_iters')
plt.ylabel('loss')
plt.show()
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
valid_hyp1 = forward(valid_x, params, 'layer1',relu)
valid_hyp2 = forward(valid_hyp1,params,'hidden1',relu)
valid_hyp3 = forward(valid_hyp2,params,'hidden2',relu)
valid_out = forward(valid_hyp3,params,'output',sigmoid)
indices = [0, 1, 300, 301, 1000, 1001, 1300, 1301, 1700, 1701]
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(0, 10, 2):
    plt.subplot(2, 2, 1)
    plt.imshow(valid_x[indices[i]].reshape(32, 32).T)
    plt.subplot(2, 2, 3)
    plt.imshow(valid_out[i].reshape(32, 32).T)
    plt.subplot(2, 2, 2)
    plt.imshow(valid_x[indices[i+1]].reshape(32, 32).T)
    plt.subplot(2, 2, 4)
    plt.imshow(valid_out[i+1].reshape(32, 32).T)
    plt.show()



##########################
##### your code here #####
##########################


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
valid_hyp1 = forward(valid_x, params, 'layer1',relu)
valid_hyp2 = forward(valid_hyp1,params,'hidden1',relu)
valid_hyp3 = forward(valid_hyp2,params,'hidden2',relu)
valid_out = forward(valid_hyp3,params,'output',sigmoid)

psnrs = 0
for i in range(valid_x.shape[0]) :
    real_out = valid_x[i]
    pred_out = valid_out[i]
    psnrs += psnr(real_out, pred_out)
print ('PSNR :',psnrs / valid_examples)
