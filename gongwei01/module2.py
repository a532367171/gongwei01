import tensorflow as tf
import numpy as np

#创建文件列表，创建文件输入队列

files = tf.train.match_filenames_once("path/to/pattern-*")
filename_queue = tf.train.string_input_producer(files,shuffle=False)
#解析TFRecord格式文件的数据
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features = tf.train.Features({
    "image":tf.FixedLenFeature([],tf.string),
    "label":tf.FixedLenFeature([],tf.int64),
    "height":tf.FixedLenFeature([],tf.int64),
    "width":tf.FixedLenFeature([],tf.int64),
    "channels":tf.FixedLenFeature([],tf.int64),
}))

image,label = features['image'],features['label']
height,width = features['height'],features['weight']
channels = features['channels']
#从原始图像解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image,tf.uint8)
decoded_image.set_shape([height,width,channels])
#定义神经网络输入层图片的大小
image_size = 299


def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image,32.0/255.0)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,32.0/255.0)
        image = tf.image.random_hue(image,0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)

    return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image,height,width,bbox):
    #如果没有标注提示框，则认为整张图想是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    #转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    #随机截取的图像，减少需要关注的物体大小对图像识别的影响
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image = tf.slice(image,bbox_begin,bbox_size)
    #将截取的图像调整为神经网络输入层大小。大小调整的算法是随机的
    distorted_image = tf.image.resize_images(distorted_image,[height,width],method=np.random.randint(4))
    #随机左右翻转图像
    distorted_image = tf.image.flip_left_right(distorted_image)
    #使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image,np.random.randint(2))
    return distorted_image


distorted_image = preprocess_for_train(decoded_image,image_size,image_size,None)

#将样例组合batch
min_after_dequeue = 10000
batch_size =100
capacity = min_after_dequeue+ batch_size *3
image_batch,label_batch = tf.train.shuffle_batch([decoded_image,label],batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)


#定义神经网络的结构及优化过程
def inference(image_data):
    """
    计算前向传播，参考前面的内容
    :param image_data:
    :return:
    """
    pass
def calc_loss(logit,label):
    """
    bp ,calc the loss,参考前面的内容
    :param logit:
    :param label:
    :return:
    """
    pass
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(10000):
        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
