import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import shutil
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import imgaug as ia


################### Load the saved data ####################

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')
num_classes = 2

################## Splitting train into train and validation ###############

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)

print("Number of training images: ", train_data.shape[0])
print("Number of validation images: ", val_data.shape[0])

################# Batch Loader ##################

## batch loader
epochs_completed = 0
idx_epoch = 0
num_train = train_data.shape[0]  ## 776

def get_next_batch(train_data, train_labels, batch_size=8):
    
    global idx_epoch
    global epochs_completed
    
    start = idx_epoch
    idx_epoch = idx_epoch + batch_size
    
    if idx_epoch > num_train:
        epochs_completed += 1
        print("Epochs completed: ", epochs_completed)
        perm = np.arange(num_train)
        np.random.shuffle(perm)
        train_images = train_data[perm]
        train_labels = train_labels[perm]
        start = 0
        idx_epoch = batch_size
        assert batch_size <= num_train
    
    end = idx_epoch
    return train_data[start:end], train_labels[start:end]

## training parameters
learning_rate = 0.0001
num_steps = 1000
batch_size = 12
display_step = 10

## network params
input_height = 48
input_width = 48
num_channels = 3
dropout = 0.5


## create wrappers

def weight_init(shape):   
    initializer = tf.contrib.layers.xavier_initializer()
    return initializer(shape)


def conv2d(x, W, b, strides=1):
    p = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    p = tf.nn.bias_add(p, b)
    return tf.nn.relu(p)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


weights = {
    'wc1': tf.Variable(weight_init([3, 3, 3, 16])),
    'wc2': tf.Variable(weight_init([3, 3, 16, 32])),
    'wd1': tf.Variable(weight_init([12 * 12 * 32, 256])),
    'out': tf.Variable(weight_init([256, num_classes]))
}

biases = {
    'bc1': tf.Variable(weight_init([16])),
    'bc2': tf.Variable(weight_init([32])),
    'bd1': tf.Variable(weight_init([256])),
    'out': tf.Variable(weight_init([num_classes]))
}


## Placeholders for feeding the data and dropout probability

x = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
y = tf.placeholder(tf.float32, shape=[None, num_classes])
keep_prob = tf.placeholder(tf.float32)


##################### The Convolutional Neural Network ######################

## define the network

def conv_net(x, weights, biases, dropout):
    
    # Tensor input is 4-D: [Batch Size, Height, Width, Channel]   
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    batch_m1, batch_var1 = tf.nn.moments(conv1,[0])
    conv1 = tf.nn.batch_normalization(conv1, batch_m1, batch_var1, variance_epsilon=1e-3, offset=None, scale=None)
    conv1 = maxpool2d(conv1)
    
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    batch_m2, batch_var2 = tf.nn.moments(conv2,[0])
    conv2 = tf.nn.batch_normalization(conv2, batch_m2, batch_var2, variance_epsilon=1e-3, offset=None, scale=None)
    conv2 = maxpool2d(conv2)
    
    ## fully connected layer
    ## reshape output of conv2 to fit FC layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    ## apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


################### Loss functions and Optimizers #####################

logits = conv_net(x, weights, biases, keep_prob)
predictions = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)


correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


#################### Training and Validation Procedure ##################

train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []

patience_cnt = 0
tol = 10


with tf.Session() as sess:
    
    sess.run(init)
    saver = tf.train.Saver()
    
    for step in range(1, num_steps+1):
        
        batch_x, batch_y = get_next_batch(train_data, train_labels, batch_size=batch_size)
        
        sess.run(optim, feed_dict={x:batch_x, y:batch_y, keep_prob:dropout})
        
        if step % display_step == 0 or step == 1:
            
            loss_train, acc_train = sess.run([loss_op, accuracy], feed_dict={x: train_data, y: train_labels, keep_prob: 1.0})
            loss_val, acc_val = sess.run([loss_op, accuracy], feed_dict={x: val_data, y: val_labels, keep_prob: 1.0})
                        
            
            print("Step " + str(step) + ", Train Loss= " + \
                  "{:.4f}".format(loss_train) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc_train) + ", Validation Loss= " + \
                 "{:.4f}".format(loss_val) + ", Validation Accuracy= " + \
                 "{:.3f}".format(acc_val))
            
            train_loss.append(loss_train)
            val_loss.append(loss_val)
            train_accuracy.append(acc_train)
            val_accuracy.append(acc_val)


        if step % display_step == 0:
            
            ## monitor val loss
            tmp = val_loss[-1] - val_loss[-2]
            if tmp >= 0:
                patience_cnt += 1
            else:
                if patience_cnt > 0:
                    patience_cnt -= 1
            
            if patience_cnt > tol:
                break
    print ("Optimization done!")


    ## Testing the model on unseen test data
    acc_test = sess.run(accuracy, feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
    print("Testing accuracy: ", acc_test) 
    ## save the model file (weights and biases)
    saver.save(sess, "./my_CNN_model")
    

    
###################### Plots ######################

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

plt.xlabel("Number of Steps", size=15)
plt.ylabel("Loss", size=15)
plt.title("Plot of Loss vs Number of Steps", size=15)

ax.plot(train_loss, 'b', label="Training Loss", linewidth=2.5)
ax.plot(val_loss, 'r', label="Validation Loss", linewidth=2.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.xaxis.labelpad=10
ax.yaxis.labelpad=10
plt.legend(loc='best',prop={'size':15})
plt.grid(True)
fig.savefig('Loss.png')



fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)


plt.xlabel("Number of Steps", size=15)
plt.ylabel("Accuracy", size=15)
plt.title("Plot of Accuracy vs Number of Steps", size=15)

ax.plot(train_accuracy, 'b', label="Training Accuracy", linewidth=2.5)
ax.plot(val_accuracy, 'r', label="Validation Accuracy", linewidth=2.5)
ax.axhline(y=acc_test, color='g', linewidth=2.5, label='Test accuracy')

trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
ax.text(0, acc_test, "{:.4f}".format(acc_test), color="g", transform=trans, 
        ha="right", va="center", size=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.xaxis.labelpad=10
ax.yaxis.labelpad=10
plt.legend(loc='best',prop={'size':15})
plt.grid(True)
fig.savefig('Accuracy.png')
