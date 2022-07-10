import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf2
import h5py
import os
import sys
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')

tf = tf2.compat.v1
tf.disable_v2_behavior()


def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_'+name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_'+name, mean)
    tf.summary.scalar(name+'_value',var)
    tf.summary.histogram('histogram_'+name, var)

def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += int(size / 2)

def segment_opp(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,77))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
            # print "x_start_end",x_train[start:end]
            # segs =  x_train[start:end]
            # segments = np.concatenate((segments,segs))
            # segments = np.vstack((segments,x_train[start:end]))
            # segments = np.vstack([segments,segs])
            # segments = np.vstack([segments,x_train[start:end]])
            # labels = np.append(labels,stats.mode(y_train[start:end]))
    return segments, labels
def segment_dap(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,9))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels

def segment_pa2(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,52))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels

def segment_sph(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,52))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels


class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def read_dataset(self):
        if self.dataset == "opp":
            path = os.path.join(os.getcwd(), "OpportunityUCIDataset", "opportunity.h5")
        elif self.dataset =="dap":
            path = os.path.join(os.getcwd(), "dataset_fog_release", 'daphnet.h5')
        elif self.dataset =="pa2":
            path = os.path.join(os.getcwd(), "PAMAP2_Dataset", "pamap2.h5")
            #path = os.path.join(os.path.expanduser('~'), 'Downloads', 'PAMAP2_Dataset', 'pamap2.h5')
        elif self.dataset =="sph":
            path = os.path.join(os.path.expanduser('~'), 'Downloads', 'SphereDataset', 'sphere.h5')
        else:
            print( "Dataset not supported yet")
            sys.exit()
        print("PATH: ", path)
        f = h5py.File(path, 'r')


        self.x_train = f.get('train').get('inputs')[()]
        self.y_train = f.get('train').get('targets')[()]

        self.x_test = f.get('test').get('inputs')[()]
        self.y_test = f.get('test').get('targets')[()]

        print("x_train shape = ", self.x_train.shape)
        print("y_train shape =", self.y_train.shape)
        print("x_test shape =", self.x_test.shape)
        print("y_test shape =", self.y_test.shape)

        return self.x_train, self.y_train, self.x_test, self.y_test


    def downsample_dataset(self):

        if self.dataset == "dap":
            # downsample to 30 Hz
            self.x_train = self.x_train[::2,:]
            self.y_train = self.y_train[::2]
            self.x_test = self.x_test[::2,:]
            self.y_test = self.y_test[::2]
            print( "x_train shape(downsampled) = ", self.x_train.shape)
            print( "y_train shape(downsampled) =",self.y_train.shape)
            print( "x_test shape(downsampled) =" ,self.x_test.shape)
            print( "y_test shape(downsampled) =",self.y_test.shape)

        if self.dataset == "pa2":
            # downsample to 30 Hz
            self.x_train = self.x_train[::3,:]
            self.y_train = self.y_train[::3]
            self.x_test = self.x_test[::3,:]
            self.y_test = self.y_test[::3]
            print( "x_train shape(downsampled) = ", self.x_train.shape)
            print( "y_train shape(downsampled) =",self.y_train.shape)
            print( "x_test shape(downsampled) =" ,self.x_test.shape)
            print( "y_test shape(downsampled) =",self.y_test.shape)

        print( np.unique(self.y_train))
        print( np.unique(self.y_test))
        unq = np.unique(self.y_test)

        return self.x_train, self.y_train, self.x_test, self.y_test


    def segment_dataset(self):
        if self.dataset == "opp":
            self.input_width = 23
            print( "segmenting signal...")
            self.train_x, self.train_y = segment_opp(self.x_train,self.y_train,self.input_width)
            self.test_x, self.test_y = segment_opp(self.x_test,self.y_test,self.input_width)
            print( "signal segmented.")
        elif self.dataset =="dap":
            print( "dap seg")
            self.input_width = 25
            print( "segmenting signal...")
            self.train_x, self.train_y = segment_dap(self.x_train,self.y_train,self.input_width)
            self.test_x, self.test_y = segment_dap(self.x_test,self.y_test,self.input_width)
            print( "signal segmented.")
        elif self.dataset =="pa2":
            self.input_width = 25
            print( "segmenting signal...")
            self.train_x, self.train_y = segment_pa2(self.x_train,self.y_train,self.input_width)
            self.test_x, self.test_y = segment_pa2(self.x_test,self.y_test,self.input_width)
            print( "signal segmented.")
        elif self.dataset =="sph":
            self.input_width = 25
            print( "segmenting signal...")
            self.train_x, self.train_y = segment_sph(self.x_train,self.y_train,self.input_width)
            self.test_x, self.test_y = segment_sph(self.x_test,self.y_test,self.input_width)
            print( "signal segmented.")
        else:
            print( "no correct dataset")


        print( "train_x shape =",self.train_x.shape)
        print( "train_y shape =",self.train_y.shape)
        print( "test_x shape =",self.test_x.shape)
        print( "test_y shape =",self.test_y.shape)

        return self.train_x, self.train_y, self.test_x, self.test_y

    def get_final(self):
        train = pd.get_dummies(self.train_y)
        test = pd.get_dummies(self.test_y)

        train, test = train.align(test, join='inner', axis=1) # maybe 'outer' is better

        self.train_y = np.asarray(train)
        self.test_y = np.asarray(test)


        print( "unique test_y",np.unique(self.test_y))
        print( "unique train_y",np.unique(self.train_y))
        print( "test_y[1]=",self.test_y[1])
        # test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)
        print( "train_y shape(1-hot) =",self.train_y.shape)
        print( "test_y shape(1-hot) =",self.test_y.shape)

class Model:
    def __init__(self, dataset_obj):
        self.dataset_obj = dataset_obj
    def build_model(self, load_weights=False, self_training=False, num_transforms=2):
        self.load_weights = load_weights
        self.self_training = self_training
        self.num_transforms = num_transforms
        self.get_model_params()
        self.get_model()
        self.get_data_preparation()
    def weight_variable(self, shape, name):
        if name == "W_conv2" and self.load_weights:
            initial = tf.constant(self.W_conv2_weight)
            print("WEIGHTS ARE LOADED")
        else:
            initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        if name == "b_conv2" and self.load_weights:
            initial = tf.constant(self.b_conv2_weight)
            print("WEIGHTS ARE LOADED")
        else:
            initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def depth_conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool(self, x, kernel_size, stride_size):
        return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                              strides=[1, 1, stride_size, 1], padding='VALID')
    def get_model_params(self):
        if self.dataset_obj.dataset=="opp":
            print( "opp")
            self.input_height = 1
            self.input_width = self.dataset_obj.input_width #or 90 for actitracker
            self.num_labels = 18  #or 6 for actitracker
            self.num_channels = 77 #or 3 for actitracker
        elif self.dataset_obj.dataset=="dap":
            print( "dap")
            self.input_height = 1
            self.input_width = self.dataset_obj.input_width #or 90 for actitracker
            self.num_labels = 2  #or 6 for actitracker
            self.num_channels = 9 #or 3 for actitracker
        elif self.dataset_obj.dataset == "pa2":
            print( "pa2")
            self.input_height = 1
            self.input_width = self.dataset_obj.input_width #or 90 for actitracker
            self.num_labels = 11  #or 6 for actitracker
            self.num_channels = 52 #or 3 for actitracker
        elif dataset_obj.dataset =="sph":
            print( "sph")
            self.input_height = 1
            self.input_width = self.dataset_obj.input_width #or 90 for actitracker
            self.num_labels = 20  #or 6 for actitracker
            self.num_channels = 52 #or 3 for actitracker
        else:
            print( "wrong dataset")
        if self.self_training:
            self.num_labels = self.num_transforms

    def get_model(self):
        stride_size = 2
        kernel_size_1 = 7
        kernel_size_2 = 3
        depth_1 = 128
        depth_2 = 128
        num_hidden = 512 # neurons in the fully connected layer
        self.dropout_1 = tf.placeholder(tf.float32) #0.1
        self.dropout_2 = tf.placeholder(tf.float32) #0.25
        self.dropout_3 = tf.placeholder(tf.float32) #0.5


        self.X = tf.placeholder(tf.float32, shape=[None,self.input_height,self.input_width,self.num_channels])
        self.Y = tf.placeholder(tf.float32, shape=[None,self.num_labels])

        print( "X shape =",self.X.shape)
        print( "Y shape =",self.Y.shape)


        # hidden layer 1
        self.W_conv1 = self.weight_variable([1, kernel_size_1, self.num_channels, depth_1], "W_conv1")
        self.b_conv1 = self.bias_variable([depth_1], "b_conv1")

        h_conv1 = tf.nn.relu(self.depth_conv2d(self.X, self.W_conv1) + self.b_conv1)
        # h_conv1 = tf.nn.dropout(tf.identity(h_conv1), dropout_1)
        h_conv1 = tf.nn.dropout(h_conv1, self.dropout_1)

        h_pool1 = self.max_pool(h_conv1,kernel_size_1,stride_size)

        print( "hidden layer 1 shape")
        print( "W_conv1 shape =",self.W_conv1.get_shape())
        print( "b_conv1 shape =",self.b_conv1.get_shape())
        print( "h_conv1 shape =",h_conv1.get_shape())
        print( "h_pool1 shape =",h_pool1.get_shape())


        # hidden layer 2
        self.W_conv2 = self.weight_variable([1, kernel_size_2, depth_1, depth_2], name="W_conv2")
        self.b_conv2 = self.bias_variable([depth_2], "b_conv2")

        h_conv2 = tf.nn.relu(self.depth_conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_conv2 = tf.nn.dropout(h_conv2, self.dropout_2)

        h_pool2 = self.max_pool(h_conv2,kernel_size_2,stride_size)



        print( "hidden layer 2 shape")
        print( "W_conv2 shape =",self.W_conv2.get_shape())
        print( "b_conv2 shape =",self.b_conv2.get_shape())
        print( "h_conv2 shape =",h_conv2.get_shape())
        print( "h_pool2 shape =",h_pool2.get_shape())


        # fully connected layer

        #first we get the shape of the last layer and flatten it out
        shape = h_pool2.get_shape().as_list()
        print( "shape's shape:", shape)

        self.W_fc1 = self.weight_variable([shape[1] * shape[2] * shape[3],num_hidden], name="W_fc1")
        self.b_fc1 = self.bias_variable([num_hidden], name="b_fc1")

        h_pool3_flat = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])
        print( "c_flat shape =",h_pool3_flat.shape)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,self.W_fc1) + self.b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, self.dropout_3)


        #readout layer.

        self.W_fc2 = self.weight_variable([num_hidden,self.num_labels], name="W_fc2")
        self.b_fc2 = self.bias_variable([self.num_labels], name="b_fc2")

        self.y_conv = tf.matmul(h_fc1,self.W_fc2) + self.b_fc2

    def get_data_preparation(self):
        train = pd.get_dummies(self.dataset_obj.train_y)
        test = pd.get_dummies(self.dataset_obj.test_y)

        train, test = train.align(test, join='inner', axis=1)  # maybe 'outer' is better

        self.train_y = np.asarray(train)
        self.test_y = np.asarray(test)

        print("unique test_y", np.unique(self.test_y))
        print("unique train_y", np.unique(self.train_y))
        print("test_y[1]=", self.test_y[1])
        # test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)
        print("train_y shape(1-hot) =", self.train_y.shape)
        print("test_y shape(1-hot) =", self.test_y.shape)
        self.train_x = self.dataset_obj.train_x.reshape(len(self.dataset_obj.train_x),1,self.input_width,self.num_channels) # opportunity
        self.test_x = self.dataset_obj.test_x.reshape(len(self.dataset_obj.test_x),1,self.input_width,self.num_channels) # opportunity
        print( "train_x_reshaped = " , self.train_x.shape)
        print( "test_x_reshaped = " , self.test_x.shape)
        print( "train_x shape =",self.dataset_obj.train_x.shape)
        print( "train_y shape =",self.dataset_obj.train_y.shape)
        print( "test_x shape =",self.dataset_obj.test_x.shape)
        print( "test_y shape =",self.dataset_obj.test_y.shape)

        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def run_model(self, training_epochs=50):

        # DEFINING THE MODEL
        learning_rate = 0.0001
        batch_size = 64
        total_batches = self.dataset_obj.train_x.shape[0] // batch_size

        # COST FUNCTION
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.y_conv))#-tf.reduce_sum(Y * tf.log(y_conv))
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # TRAINING THE MODEL
        loss_over_time = np.zeros(training_epochs)
        saver = tf.train.Saver()
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            train_flag = True
            if train_flag:
                for epoch in range(training_epochs):
                    cost_history = np.empty(shape=[0],dtype=float)
                    for b in range(total_batches):
                        offset = (b * batch_size) % (self.train_y.shape[0] - batch_size)
                        batch_x = self.train_x[offset:(offset + batch_size), :, :, :]
                        batch_y = self.train_y[offset:(offset + batch_size), :]

                        _, c= session.run([optimizer, loss],feed_dict={self.X: batch_x, self.Y : batch_y, self.dropout_1: 1-0.1, self.dropout_2: 1-0.25, self.dropout_3: 1-0.5})
                        cost_history = np.append(cost_history,c)
                    print( "Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",session.run(
                        accuracy, feed_dict={self.X: self.train_x, self.Y: self.train_y, self.dropout_1: 1-0.1, self.dropout_2: 1-0.25, self.dropout_3: 1-0.5}))
                    loss_over_time[epoch] = np.mean(cost_history)
            else:
                saver.restore(session, self.dataset_obj.dataset)
            self.save_weights()
            saver.save(session, self.dataset_obj.dataset)

    def save_weights(self):
        self.W_conv2_weight = self.W_conv1.eval()
        self.b_conv2_weight = self.b_conv1.eval()
        self.W_conv2_weight = self.W_conv2.eval()
        self.b_conv2_weight = self.b_conv2.eval()
        self.W_fc1_weight = self.W_fc1.eval()
        self.b_fc1_weight = self.b_fc1.eval()


    def test(self):
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, self.dataset_obj.dataset)
            test_accuracy = session.run(self.accuracy, feed_dict={self.X: self.test_x,
                                                                  self.Y: self.test_y,
                                                                  self.dropout_1: 1,
                                                                  self.dropout_2: 1,
                                                                  self.dropout_3: 1})
            print( "Testing Accuracy:", test_accuracy)
            y_p = tf.argmax(self.y_conv, 1)
            val_accuracy, y_pred = session.run([self.accuracy, y_p], feed_dict={self.X:self.test_x,
                                                                                self.Y:self.test_y,
                                                                                self.dropout_1: 1,
                                                                                self.dropout_2: 1,
                                                                                self.dropout_3: 1})
            print( "validation accuracy:", val_accuracy)
            y_true = np.argmax(self.test_y, 1)
            return test_accuracy, val_accuracy, y_pred, y_true



    def self_train(self, transforms, training_epochs=50):

        # DEFINING THE MODEL


        learning_rate = 0.0001
        #training_epochs = 2
        batch_size = 64
        total_batches = self.dataset_obj.train_x.shape[0] // batch_size

        # COST FUNCTION
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.y_conv))#-tf.reduce_sum(Y * tf.log(y_conv))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # TRAINING THE MODEL
        loss_over_time = np.zeros(training_epochs)

        saver = tf.train.Saver()
        with tf.Session() as session:
            tf.initialize_all_variables().run()

            indexes = get_random_spilt_one_hot(self.train_x.shape[0], self.num_labels)
            for i in range(len(indexes)):
                if np.all(indexes[i] != get_one_hot(0, self.num_labels)):
                    before_transform = self.train_x[i].copy()
                    to_transform = np.swapaxes(self.train_x[i, 0, :, :], 0, 1)
                    to_transform = transforms[get_class_from_one_hot(indexes[i])](to_transform)
                    to_transform = np.swapaxes(to_transform, 0, 1)
                    to_transform = np.reshape(to_transform, (1, to_transform.shape[0], to_transform.shape[1]))
                    self.train_x[i] = to_transform
                    if np.all(to_transform == before_transform):
                        print("SELF-TRANSFORM")
            self.train_y = indexes

            for epoch in range(training_epochs):
                cost_history = np.empty(shape=[0],dtype=float)
                for b in range(total_batches):
                    offset = (b * batch_size) % (self.train_y.shape[0] - batch_size)
                    batch_x = self.train_x[offset:(offset + batch_size), :, :, :]
                    batch_y = self.train_y[offset:(offset + batch_size), :]

                    _, c= session.run([optimizer, loss],feed_dict={self.X: batch_x, self.Y : batch_y, self.dropout_1: 1-0.1, self.dropout_2: 1-0.25, self.dropout_3: 1-0.5})
                    cost_history = np.append(cost_history,c)
                print( "Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",session.run(
                    accuracy, feed_dict={self.X: self.train_x, self.Y: self.train_y, self.dropout_1: 1-0.1, self.dropout_2: 1-0.25, self.dropout_3: 1-0.5}))
                loss_over_time[epoch] = np.mean(cost_history)

            self.save_weights()
            saver.save(session, self.dataset_obj.dataset)

def get_random_spilt_one_hot(num_samples, num_classes):
    split = int(num_samples / num_classes)
    indexes = []
    for i in range(num_classes):
        indexes += [get_one_hot(i, num_classes) for j in range(split)]
    for i in range (num_samples % num_classes):
        indexes += [get_one_hot(i, num_classes)]
    random.shuffle(indexes)
    return np.array(indexes)

def get_one_hot(i, num_classes):
    one_hot = [0 for j in range(num_classes)]
    one_hot[i] = 1
    return np.array(one_hot)

def get_class_from_one_hot(one_hot):
    return np.where(one_hot == 1)[0][0]
