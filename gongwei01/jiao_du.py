import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from math import pi
 
'''按角度旋转图片'''
 
 
def rotate_images(X_imgs, start_angle, end_angle, n_images,image_size_w,image_size_h):
     X_rotate = []
     iterate_at = (end_angle - start_angle) / (n_images - 1)
 
     tf.reset_default_graph()
     X = tf.placeholder(tf.float32, shape=(None, image_size_w,image_size_h, 3))
     radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
     tf_img = tf.contrib.image.rotate(X, radian)
     with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
 
          for index in range(n_images):
               degrees_angle = start_angle + index * iterate_at
               radian_value = degrees_angle * pi / 180  # Convert to radian
               radian_arr = [radian_value] * len(X_imgs)
               rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
               X_rotate.extend(rotated_imgs)
 
     X_rotate = np.array(X_rotate, dtype=np.float32)
     return X_rotate
 
#image = cv2.imread('data\\train\\全.全 192.168.1.65_01_20190307085134847_55_00011630.jpg')
##image = tf.gfile.GFile('/home/ubuntu/images/output.jpg', 'wb'
#b,g,r = cv2.split(image)
#image = cv2.merge([r,g,b])
#image = cv2.resize(image, (200,200),0,0, cv2.INTER_LINEAR)  #图像缩放
#image = np.multiply(image, 1.0 / 255.0)



picpath = 'data\\train\\全.全 192.168.1.65_01_20190307085134847_55_00011630.jpg'
    # 加载图片，并解码得到三维数组, 注意打开模式必须是rb
#raw_pic = tf.gfile.FastGFile(picpath, 'rb').read()
#    # 解码得到三维数组
#raw_data = tf.image.decode_jpeg(raw_pic)


image = tf.read_file(picpath)
image = tf.image.decode_jpeg(image, channels=3)  # 这里是jpg格式
image = tf.image.resize_images(image, [208, 208])
image = tf.cast(image, tf.float32) / 255.  # 转换数据类型并归一化



list_image = []
list_image.append(image)
list_image = np.array(list_image)
#scaled_imgs = central_scale_images(list_image, [1.30, 1.50, 1.80])
scaled_imgs = rotate_images(list_image, -90, 90, 5,200,200)
#image = scaled_imgs[0]/255.
 
plt.subplot(331), plt.imshow(image), plt.title('', fontsize='medium')
plt.subplot(332), plt.imshow(scaled_imgs[0]), plt.title('',fontsize='small')
plt.subplot(333), plt.imshow(scaled_imgs[1]), plt.title('', fontsize='small')
plt.subplot(334), plt.imshow(scaled_imgs[2]), plt.title('', fontsize='small')
plt.subplot(335), plt.imshow(scaled_imgs[3]), plt.title('', fontsize='small')
plt.subplot(336), plt.imshow(scaled_imgs[4]), plt.title('', fontsize='small')
#plt.subplot(337), plt.imshow(scaled_imgs[5]), plt.title('', fontsize='small')
#plt.subplot(258), plt.imshow(scaled_imgs[6]), plt.title('Translate bottom 20 percent', fontsize='medium')
plt.show()
