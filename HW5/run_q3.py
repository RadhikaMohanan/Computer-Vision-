import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 150
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.002
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')

init_HW = params['Wlayer1']

# with default settings, you should get loss < 150 and accuracy > 80%
train_loss_dat = np.zeros(max_iters)
valid_loss_dat = np.zeros(max_iters)
train_acc_dat = np.zeros(max_iters)
valid_acc_dat = np.zeros(max_iters)


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    count = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        count += 1
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        loss,acc = compute_loss_and_acc(yb, probs)
        total_loss +=  loss
        total_acc += acc
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        
        params['Wlayer1'] = params['Wlayer1']- learning_rate * params['grad_Wlayer1']
        params['Woutput'] = params['Woutput'] - learning_rate * params['grad_Woutput']
        params['blayer1'] = params['blayer1'] - learning_rate * params['grad_blayer1']
        params['boutput'] = params['boutput'] - learning_rate * params['grad_boutput']

    total_acc = total_acc / batch_num
    total_loss = total_loss / train_x.shape[0]
    train_loss_dat[itr] = total_loss
    train_acc_dat[itr] = total_acc

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    
    h1_valid = forward(valid_x,params,'layer1')
    probs_valid = forward(h1_valid,params,'output',softmax)
    v_l,v_a = compute_loss_and_acc(valid_y, probs_valid)
    valid_loss_dat[itr] = v_l / valid_x.shape[0]
    valid_acc_dat[itr] = v_a

plt.figure('accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.plot(range(max_iters), train_acc_dat,'r')
plt.plot(range(max_iters), valid_acc_dat,'b')
plt.legend(['train', 'validation'])
plt.show()

plt.figure('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(max_iters), train_loss_dat,'r')
plt.plot(range(max_iters), valid_loss_dat,'b')
plt.legend(['train', 'validation'])
plt.show()


# run on validation set and report accuracy! should be above 75%
h1_valid = forward(valid_x,params,'layer1')
probs_valid = forward(h1_valid,params,'output',softmax)
loss_valid,valid_acc = compute_loss_and_acc(valid_y, probs_valid)
##########################
##### your code here #####
##########################

print('Validation accuracy: ',valid_acc)


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #####
    hiddenWeights = saved_params['Wlayer1']
# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################
#initial Layer Weight
fig = plt.figure()
grid = ImageGrid(fig, 111,  
                 nrows_ncols=(8, 8),  
                 axes_pad=0.1, 
                 )

for i in range(init_HW.shape[1]):
    layerWeight = init_HW[:,i]
    layerImage = np.reshape(layerWeight, (32, 32))
    grid[i].imshow(layerImage)

plt.show()

#Hidden Layer Weight
fig = plt.figure()
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(8, 8), 
                 axes_pad=0.1,  
                 )

for i in range(hiddenWeights.shape[1]):
    layerWeight = hiddenWeights[:,i]
    layerImage = np.reshape(layerWeight, (32, 32))
    grid[i].imshow(layerImage)

plt.show()





# Q3.4


# compute comfusion matrix here
##########################
##### your code here #####
##########################
#Training data confusion Matrix
confusion_matrix_tr = np.zeros((train_y.shape[1],train_y.shape[1]))

h1_tr = forward(train_x,params,'layer1')
probs_tr = forward(h1_tr,params,'output',softmax)
true_tr_label = np.argmax(train_y, axis = 1)
predLabel_tr = np.argmax(probs_tr, axis = 1)
for i in range(true_tr_label.size) :
    confusion_matrix_tr[true_tr_label[i]][predLabel_tr[i]] += 1


plt.imshow(confusion_matrix_tr,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()


#Validation data confusion Matrix
confusion_matrix_v = np.zeros((valid_y.shape[1],valid_y.shape[1]))

h1_v = forward(valid_x,params,'layer1')
probs_v = forward(h1_v,params,'output',softmax)
true_v_Label = np.argmax(valid_y, axis = 1)
predLabel_v = np.argmax(probs_v, axis = 1)
for i in range(true_v_Label.size) :
    confusion_matrix_v[true_v_Label[i]][predLabel_v[i]] += 1


plt.imshow(confusion_matrix_v,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()