# -*- coding: utf-8 -*-
import time
#from load_data import *
from model import *
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
import numpy as np


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
                                                                      batch_size=8,
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


# 测试检查点
def eval():
    N_CLASSES = 8
    IMG_SIZE = 208
    BATCH_SIZE = 20
    CAPACITY = 200
    MAX_STEP = 10

    test_dir = 'data\\test'
    logs_dir = 'logs_1'     # 检查点目录

    sess = tf.Session()

    train_list = get_all_files(test_dir, is_random=True)
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    #train_logits1 = inference(image_train_batch, N_CLASSES)
    #train_logits2 = tf.nn.softmax(train_logits1)  # 用softmax转化为百分比数值

    #image_train_batch = tf.gfile.FastGFile('data/test/窑3.窑3 C62618904_1_20190125T130011Z_20190125T132547Z_0006500.jpg', 'r').read()  
    train_logits = inference(image_train_batch, N_CLASSES)
    #train_logits = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值

    # 载入检查点

    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            image, prediction = sess.run([image_train_batch, train_logits])


            #image_train_batch1,label_train_batch1 = sess.run(image_train_batch)

            #train_logits3= sess.run(train_logits2)
            #image_train_batch1= sess.run(image_train_batch)



            i=0

            #image, prediction = sess.run([image_train_batch1, train_logits3])
            #train_logits3= sess.run(train_logits2)
            #image_train_batch2,train_logits2 = sess.run([image_train_batch1,train_logits3])
            ##sess.run(feed_dict={images: image_train_batch1, y_: label_train_batch1})
            #image, prediction = sess.run([image_train_batch,train_logits])
            #for x1 in [prediction]:
            # for x in [x1[0]]:
            #  max_index = np.argmax(x)
            #  if max_index == 0:
            #    label = '%.2f%% is a kong' % (x[0] * 100)
            #  elif max_index == 1:
            #    label = '%.2f%% is a 1 yao' % (x[1] * 100)
            #  elif max_index == 2:
            #    label = '%.2f%% is a 2 yao' % (x[2] * 100)
            #  elif max_index == 3:
            #    label = '%.2f%% is a 3 yao' % (x[3] * 100)
            #  elif max_index == 4:
            #    label = '%.2f%% is a 4 yao' % (x[4] * 100)
            #  elif max_index == 5:
            #    label = '%.2f%% is a xian 1' % (x[5] * 100)
            #  elif max_index == 6:
            #    label = '%.2f%% is a xian 2' % (x[6] * 100)
            #  else:     # 空标记为'0'
            #    label = '%.2f%% is a quan' % (x[7] * 100)


            #  plt.imshow(image[i])
            #  plt.title(label)
            #  plt.show()
            #  i=i+1
            #  pass
            # pass




            #    pass
            #for x in [prediction]:
            for x in range(len(prediction)):
             max_index = np.argmax(prediction[x])
             if max_index == 0:
                label = '%.2f%% is a kong' % (prediction[x][0] * 100)
             elif max_index == 1:
                label = '%.2f%% is a 1 yao' % (prediction[x][1] * 100)
             elif max_index == 2:
                label = '%.2f%% is a 2 yao' % (prediction[x][2] * 100)
             elif max_index == 3:
                label = '%.2f%% is a 3 yao' % (prediction[x][3] * 100)
             elif max_index == 4:
                label = '%.2f%% is a 4 yao' % (prediction[x][4] * 100)
             elif max_index == 5:
                label = '%.2f%% is a xian 1' % (prediction[x][5] * 100)
             elif max_index == 6:
                label = '%.2f%% is a xian 2' % (prediction[x][6] * 100)
             else:     # 空标记为'0'
                label = '%.2f%% is a quan' % (prediction[x][7] * 100)
             plt.imshow(image[i])
             plt.title(label)
             plt.show()
             i=i+1
            # pass

           

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()

        #'''
    #:param input_checkpoint:
    #:param output_graph: PB模型保存路径
    #:return:
    #'''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点

#def freeze_graph(input_checkpoint,output_graph):
#    output_node_names = "softmax_linear"
#    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#    graph = tf.get_default_graph() # 获得默认的图
#    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
#    with tf.Session() as sess:
#        saver.restore(sess, input_checkpoint) #恢复图并得到数据
#          # 模型持久化，将变量值固定
#          # 等于:sess.graph_def
#          # 如果有多个输出节点，以逗号隔开
#        output_graph_def = graph_util.convert_variables_to_constants(
#            sess=sess,
#            input_graph_def=input_graph_def,
#            output_node_names=output_node_names.split(","))
 
#        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
#            f.write(output_graph_def.SerializeToString()) #序列化输出
#        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
 
#        # for op in graph.get_operations():
#        #     print(op.name, op.values())

def freeze_graph(input_checkpoint,output_graph):
    #checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    #input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "softmax_z/softmax_z"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            print("----------------------------------------------------------------")
            f.write(output_graph_def.SerializeToString()) #序列化输出
            print("----------------------------------------------------------------")
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        
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

            plt.imshow(im1)
            #plt.title(label)
            plt.show()


            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print("pre class_id:{}".format(sess.run(class_id)))
            return str(max_index)



if __name__ == '__main__':
     training()
     #eval()
     #freeze_graph('logs_3\\model.ckpt-33000','logs_3\\pb\\frozen_model_33000_04_07.pb')
     #freeze_graph_test('logs_3\\pb\\frozen_model1.pb','data\\test1\\1.jpg')
     #freeze_graph_test('D:\\vs2017\\出窑服务图片识别版\\出窑服务图片识别版\\出窑服务图片识别版\\bin\\x64\\Debug\\logs_2\\pb\\frozen_model12.pb','data\\test1\\1.jpg')

