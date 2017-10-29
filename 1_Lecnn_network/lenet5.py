import tensorflow as tf
import numpy as np
from load_data import load_data
import argparse
import os
import matplotlib.pyplot as plt
from lenet_utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--show_data', type=str, nargs='*', default=0, help="show the figure of several data")
parser.add_argument('--batch_size', type=int, default=128, help='the number of images of per batch')
parser.add_argument('--epochs', type=int, default = 300, help='the iteation times of the total set')
parser.add_argument('--used_data_size', type=int, default=60000, help='the number modling used')
parser.add_argument('--data_path', type=str, help='the data path')

args = parser.parse_args()
batch_size = args.batch_size
leng = args.used_data_size
epochs = args.epochs
saverpath = './model_save/lenet5/'
if not os.path.exists(saverpath):
    os.mkdir('./model_save')
    os.mkdir(saverpath)
# 读取数据
if args.data_path:
    data_path = args.data_path
else:
    data_path = os.getcwd()
    data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, 'download_data\\cifar10\\cifar-10-batches-py')
(x_train, y_train_index), (x_test, y_test_index), label = load_data(data_path)
y_train_index = y_train_index.ravel()
y_test_index = y_test_index.ravel()
y_train = np.zeros((y_train_index.shape[0], 10))
y_test = np.zeros((y_test_index.shape[0], 10))
y_train[np.arange(y_train_index.shape[0]), y_train_index] = 1
y_test[np.arange(y_test_index.shape[0]), y_test_index] = 1
if args.show_data != 0:
    for i in range(6):
        for j in range(10):
            n = np.random.randint(0, 50000, 1)
            img = x_train[n]
            img.shape = (32,32,3)
            plt.subplot(6,10,i * 10 + j + 1)
            plt.imshow(img)
            indicator = y_train_index[n]
            plt.title(label[indicator])
    plt.show()




def main():
    global x_train, y_train, x_test, y_test
    conv_filters = [[5,5,3,6], [5,5,6,16]]
    nets_cf = [120, 84, 10]
    conv_layers_out_flatten = int((((32 + 1 - 5) / 2 + 1 - 5) / 2) ** 2 * 16)
    train_batch_x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='train_batch_x')
    train_batch_y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name = 'train_batch_y')
    

     
    
    
#    reshape the input signal
    train_batch_x_tensor = tf.reshape(train_batch_x, [-1, 32, 32, 3])
#    layer 1
    W_conv1 = weight_variable(conv_filters[0], name = 'Weight_conv1')
    b_conv1 = bias_variable([conv_filters[0][3]], name = 'bias_conv1')
    o_conv1 = tf.nn.relu(tf.nn.conv2d(train_batch_x_tensor, filter=W_conv1, strides=[1,1,1,1], padding='VALID') + b_conv1)
    o_out1 = tf.nn.max_pool(o_conv1, [1,2,2,1], strides=[1,2,2,1], padding='VALID')
    tf.summary.histogram('Weight_conv1', W_conv1)
    tf.summary.histogram('bias_conv1', b_conv1)
#     layer2
    W_conv2 = weight_variable(conv_filters[1], name = 'Weight_conv2')
    b_conv2 = bias_variable([conv_filters[1][3]], name = 'bias_conv2')
    o_conv2 = tf.nn.relu(tf.nn.conv2d(o_out1, filter=W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2)
    o_out2 = tf.nn.max_pool(o_conv2, [1,2,2,1], strides=[1,2,2,1], padding='VALID')
    tf.summary.histogram('W_conv2', W_conv2)
    tf.summary.histogram('b_conv2', b_conv2)
#     full connection layer3
    o_conv3_flatten = tf.reshape(o_out2, [-1, conv_layers_out_flatten])
    W_fc3 = weight_variable([conv_layers_out_flatten, nets_cf[0]], name = 'Weight_fc3')
    b_fc3 = bias_variable([nets_cf[0]], name = 'bias_fc3')
    o_out3 = tf.nn.relu(tf.matmul(o_conv3_flatten, W_fc3) + b_fc3)
    tf.summary.histogram('W_fc3', W_fc3)
    tf.summary.histogram('b_fc3', b_fc3)
#     full connection layer4
    W_fc4 = weight_variable([nets_cf[0], nets_cf[1]], name = 'Weight_fc4')
    b_fc4 = bias_variable([nets_cf[1]], name = 'bias_fc4')
    tf.summary.histogram('W_fc4', W_fc4)
    tf.summary.histogram('b_fc4', b_fc4)
    o_out4 = tf.nn.relu(tf.matmul(o_out3, W_fc4) + b_fc4)
#       full connection layer5 and softmax
    W_fc5 = weight_variable([nets_cf[1], nets_cf[2]], name = 'Weight_fc5')
    b_fc5 = bias_variable([nets_cf[2]], name = 'bias_fc5')
    tf.summary.histogram('W_fc5', W_fc5)
    tf.summary.histogram('b_fc5', b_fc5)
    o_out5 = tf.nn.softmax(tf.matmul(o_out4, W_fc5) + b_fc5)
#     loss function
    cross_entropy = -tf.reduce_sum(train_batch_y * tf.log(o_out5 + 1e-8))
    tf.summary.scalar(name="loss", tensor=cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
#       monitor acc
    correct_prediction = tf.equal(tf.argmax(o_out5, 1), tf.argmax(train_batch_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')
    tf.summary.scalar(name='accuracy', tensor=accuracy)
#       init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if os.path.exists(saverpath + 'checkpoint'):
        saver.restore(sess, tf.train.latest_checkpoint(saverpath))

    writer = tf.summary.FileWriter(logdir='./tensorboards/lenet5/', graph=sess.graph)
    merged_op = tf.summary.merge_all()

    tf.summary.merge_all()
    for epoch_i in range(epochs):
        chaos_index = np.random.permutation(list(range(x_train.shape[0])))
        x_train = x_train[chaos_index, :, :]
        y_train = y_train[chaos_index, :]
        for batch_i in range(leng // batch_size):
            batch_x = x_train[batch_i * batch_size : (batch_i + 1) * batch_size, :, :, :]
            batch_y = y_train[batch_i * batch_size : (batch_i + 1) * batch_size, :]

            sess.run(optimizer, feed_dict={
                train_batch_x: batch_x,
                train_batch_y: batch_y
            })

        acc, merge_res = sess.run([accuracy, merged_op], feed_dict={
            train_batch_x: x_test,
            train_batch_y: y_test
        })
        print('the acc is %f' % acc)
        writer.add_summary(merge_res, epoch_i)
        if (epoch_i % 10 == 0):
            saver.save(sess, saverpath + 'lenet5.model.ckpt', global_step=epoch_i)


if __name__ == '__main__':
    main()
    


