# -*- coding: utf-8 -*-
import time
#from load_data import *
from model import *
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
import numpy as np
from load_data import *


# 训练模型
def training():
    N_CLASSES = 8
    IMG_SIZE = 208
    BATCH_SIZE = 20
    CAPACITY = 200
    MAX_STEP = 100000
    LEARNING_RATE = 1e-4

    # 测试图片读取
    image_dir = 'data\\train'
    logs_dir = 'logs_3'     # 检查点保存路径

    sess = tf.Session()
    # 导入图片
    train_list = get_all_files(image_dir, True)
    # 导入批次                                     # train_list 训练集  IMG_SIZE 归一后的尺寸 BATCH_SIZE 一批多少张 CAPACITY 队列容量
    #image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True) 

    intput_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # 这里是jpg格式
    image_train = tf.image.resize_images(image_train, [208, 208])
    image_train = tf.cast(image_train, tf.float32) / 255.  # 转换数据类型并归一化

    # 图片标签
    label_train = intput_queue[1]

    # 获取批次
    image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train],
                                                                      batch_size=BATCH_SIZE,
                                                                      capacity=200,
                                                                      min_after_dequeue=100,
                                                                      num_threads=2)
    



    # 拿  数据+模型  训练
    #train_logits = inference(image_train_batch, N_CLASSES)
    
    
    images = tf.placeholder(tf.float32, [None,208,208,3],name='x_input')
    y_ = tf.placeholder(tf.int32, [None],name='y_')

    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1 
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    # Dropout  
    norm1 = tf.nn.dropout(norm1, 0.8)  

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")



    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
    
        # Dropout  
    norm2 = tf.nn.dropout(norm2, 0.8)  

    # full-connect1
    with tf.variable_scope("fc1") as scope:
        reshape = layers.flatten(norm2)
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 8],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[8],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")

    with tf.variable_scope("softmax_z") as scope:
        softmax= tf.nn.softmax(softmax_linear, name="softmax_z")
   # 计算损失率
    #train_loss = losses(softmax_linear, label_train_batch)

    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear,
                                                                       labels=y_)
        loss = tf.reduce_mean(cross_entropy)



   # 计算准确率
    #train_acc = evaluation(softmax_linear, label_train_batch)
    with tf.variable_scope("accuracy"):
        correct = tf.nn.in_top_k(softmax_linear, y_, 1) #tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，tf.nn.in_top_k(prediction, target, K):
        correct = tf.cast(correct, tf.float16)#将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 
        accuracy = tf.reduce_mean(correct)#根据给出的axis在input_tensor上求平均值。除非keep_dims为真，axis中的每个的张量秩会减少1。如果keep_dims为真，求平均值的维度的长度都会保持为1.如果不设置axis，所有维度上的元素都会被求平均值，并且只会返回一个只有一个元素的张量。

    # 优化器
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    #返回的是需要训练的变量列表
    var_list = tf.trainable_variables()

    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])

    print('参数数目:%d' % sess.run(paras_count), end='\n\n')

    saver = tf.train.Saver() 

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    
    #coord = tf.train.Coordinator() #创建一个协调器，管理线程
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)#启动QueueRunner, 此时文件名队列已经进队

    s_t = time.time()
     #开搞
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            image_train_batch1,label_train_batch1 = sess.run([image_train_batch,label_train_batch])
            sess.run(train_op,feed_dict={images: image_train_batch1, y_: label_train_batch1})

            #acc = sess.run([train_acc]) 

            if step % 100 == 0:  # 实时记录训练过程并显示
                accuracy_result  =sess.run(accuracy,feed_dict={images: image_train_batch1, y_: label_train_batch1})
                loss_result =sess.run(loss,feed_dict={images: image_train_batch1, y_: label_train_batch1})
                runtime = time.time() - s_t
                print('Step: %6d, loss: %.8f, accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, loss_result, accuracy_result * 100, runtime, (MAX_STEP - step) * runtime / 360000))
                #print('Step: %6d, loss: %.8f,  accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                #      % (step,loss, acc * 100, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()
                #print("----------------------------------------------------------------")
                #graph_def = sess.graph.as_graph_def()
                #ops = sess.graph.get_operations()
                #for op in ops:
                # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # print(op.name, op.outputs)
                # print("----------------------------------------------------------------")
            if step % 3000 == 0 or step == MAX_STEP - 1:  # 保存检查点
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()






if __name__ == '__main__':
     training()