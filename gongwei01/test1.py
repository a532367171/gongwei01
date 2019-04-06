# -*- coding: utf-8 -*-

import sys
sys.path.append("C:\\ProgramData\\Anaconda3\\envs\\tensorflow_python368\\Lib");
sys.path.append("C:\\ProgramData\\Anaconda3\\envs\\tensorflow_python368\\Lib\\site-package");

import time
#from load_data import *
#from model import *
#import matplotlib.pyplot as plt
#from tensorflow.python.framework import graph_util
import tensorflow as tf

import numpy as np




def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    #im = tf.read_file(image_path)
    #im = tf.image.decode_jpeg(im, channels=3)  # 这里是jpg格式
    #im = tf.image.resize_images(im, [208, 208])
    #im = tf.cast(im, tf.float32) / 255.  # 转换数据类型并归一化
    with tf.Graph().as_default():
        im = tf.read_file(image_path)
        im = tf.image.decode_jpeg(im, channels=3)  # 这里是jpg格式
        im = tf.image.resize_images(im, [208, 208])
        im = tf.cast(im, tf.float32) / 255.  # 转换数据类型并归一化
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            im1= sess.run(im)
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("x_input:0")
            #input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            #input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
 
            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("softmax_z/softmax_z:0")
 
            # 读取测试图片

            # 从路径中读取图片

            #im = tf.read_file(image_path)
            #im = tf.image.decode_jpeg(im, channels=3)  # 这里是jpg格式
            #im = tf.image.resize_images(im, [208, 208])
            #im = tf.cast(im, tf.float32) / 255.  # 转换数据类型并归一化

            #plt.imshow(im)
            #plt.title(label)
            #plt.show()
            #im = tf.tuple(im)
            #im = tf.reshape(im, shape=[1, im.get_shape()[0].value, im.get_shape()[1].value, im.get_shape()[2].value])

            #label_train = tf.random_normal([8])
            #image_train_batch, label_train_batch = tf.train.shuffle_batch([im, label_train],
            #                                                          batch_size=8,
            #                                                          capacity=200,
            #                                                          min_after_dequeue=100,
            #                                                          num_threads=2)

            #im=read_image(image_path,resize_height,resize_width,normalization=True)

            im2=im1[np.newaxis,:]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            #out=sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
            #                                            input_keep_prob_tensor:1.0,
            #                                            input_is_training_tensor:False})
            out=sess.run(output_tensor_name, feed_dict={input_image_tensor: im2})
            #out1= tf.nn.softmax(out)
            #out2= sess.run(out1)
            max_index = np.argmax(out)
            #if max_index == 0:
            #    label = '%.2f%% is a kong' % (out2[0][0] * 100)
            #elif max_index == 1:
            #    label = '%.2f%% is a 1 yao' % (out2[0][1] * 100)
            #elif max_index == 2:
            #    label = '%.2f%% is a 2 yao' % (out2[0][2] * 100)
            #elif max_index == 3:
            #    label = '%.2f%% is a 3 yao' % (out2[0][3] * 100)
            #elif max_index == 4:
            #    label = '%.2f%% is a 4 yao' % (out2[0][4] * 100)
            #elif max_index == 5:
            #    label = '%.2f%% is a xian 1' % (out2[0][5] * 100)
            #elif max_index == 6:
            #    label = '%.2f%% is a xian 2' % (out2[0][6] * 100)
            #else:     # 空标记为'0'
            #    label = '%.2f%% is a quan' % (out2[0][7] * 100)

            #plt.imshow(im1)
            #plt.title(max_index)
            #plt.show()


            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print("pre class_id:{}".format(sess.run(class_id)))
            return str(max_index)



if __name__ == '__main__':
     freeze_graph_test('logs_3\\pb\\frozen_model1.pb','data\\test1\\1.jpg')