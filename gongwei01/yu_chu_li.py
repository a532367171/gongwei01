# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:14:42 2018
@author: Administrator
"""
# 图像预处理的完整流程
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#from torchvision import transforms

 
# 转换图像的色彩(包括色度，饱和度，对比度， 亮度)，不同的顺序会得到不同的结果
# 转换的顺序随机
def distort_color(input_img, color_order=0):
    # 调整色度和饱和度的图像必须要求是rgb三通道图像
    # 所以要确保你的图像位深度是24位
    if color_order == 0:
        # 随机调整色度
        img = tf.image.random_hue(input_img, 0.2)
        # 随机调整饱和度
        img = tf.image.random_saturation(img, 0.8, 1.2)
        # 随机调整对比度
        img = tf.image.random_contrast(img, 0.8, 1.2)
        # 随机调整亮度
        img = tf.image.random_brightness(img, 0.2)
    elif color_order == 1:
        # 随机调整色度
        img = tf.image.random_hue(input_img, 0.2)
        # 随机调整对比度
        img = tf.image.random_contrast(img, 0.8, 1.2)
        # 随机调整亮度
        img = tf.image.random_brightness(img, 0.2)
        # 随机调整饱和度
        img = tf.image.random_saturation(img,0.8, 1.2)
    elif color_order == 2:
        # 随机调整饱和度
        img = tf.image.random_saturation(input_img, 0.8, 1.2)
        # 随机调整亮度
        img = tf.image.random_brightness(img, 0.2)
        # 随机调整色度
        img = tf.image.random_hue(input_img, 0.2)
        # 随机调整对比度
        img = tf.image.random_contrast(img,0.8, 1.2)
    image = tf.clip_by_value(img, 0.0, 1.0)
    return image
 
# 图像预处理函数
# 输入一张解码后的图像，目标图像的尺寸以及图像上的标注框
def image_preprocessing(input_img, height, width, bbox):
    ## 如果没有输入边界框，则默认整个图像都需要关注
    #if bbox is None:
    #    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],shape=(1,1,4))
    ## 转换图像的数据类型
    #if input_img.dtype != tf.float32:
    #    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    ## 随机截取图像, 减少需要关注的物体的大小对识别算法的影响
    ## 随机生成一个bounding box(大小一定,位置随机)
    #bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
    #        tf.shape(input_img), bbox)
    ## 得到随机截取的图像.
    #distorted_img = tf.slice(input_img, bbox_begin, bbox_size)
    
    ## 将随机截取的图像调整到指定大小，内插算法是随机选择的
    #distorted_img = tf.image.resize_images(distorted_img, (height, width), 
    #                                       method = np.random.randint(4))
    ## 随机左右翻转图像
    #distorted_img = tf.image.random_flip_left_right(distorted_img)
    ## 随机上下翻转图像
    #distorted_img = tf.image.random_flip_up_down(distorted_img)
    # 随机打乱图像的色彩
    distorted_img = distort_color(input_img, 
                                  color_order=np.random.randint(3))
    return distorted_img
 
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


# 定义主函数
def main():
    # 注意路径中不能有中文字符
    picpath = 'data\\train\\全.全 192.168.1.65_01_20190307085134847_55_00011630.jpg'
    # 加载图片，并解码得到三维数组, 注意打开模式必须是rb
    raw_pic = tf.gfile.FastGFile(picpath, 'rb').read()
    # 解码得到三维数组
    raw_data = tf.image.decode_jpeg(raw_pic)


    im = tf.read_file(picpath)
    im = tf.image.decode_jpeg(im, channels=3)  # 这里是jpg格式
    im = tf.image.resize_images(im, [208, 208])
    im = tf.cast(im, tf.float32) / 255.  # 转换数据类型并归一化

    # print(raw_data.get_shape())
    # 设置bounding box 大小
    bbox = tf.constant([0.2, 0.2, 0.8, 0.8], shape=(1,1,4))
    plt.figure(figsize=(8,6))
    with tf.Session() as sess:
        # 随机6次截取
        for i in range(20):
            plt.subplot(4,5,i+1)
            croped_img = image_preprocessing(im, 256, 256, bbox)

            new_img = croped_img.eval()  # 数组
            # 6.图片保存
            with tf.gfile.GFile("dog_new.png", "wb") as f:
              f.write(new_img)

            #plt_img=tensor_to_PIL(croped_img)

            #Y=sess.run(plt_img)

            #plt_img.save('02.jpg')
            #plt.savefig("filename.png")
            print(tf.shape(croped_img))
            plt.imshow(sess.run(new_img))
            plt.axis('off')
    plt.show()
 
if __name__ == '__main__':
    main()
