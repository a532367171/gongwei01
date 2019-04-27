import Augmentor
 
# 图像所在目录
AUGMENT_SOURCE_DIR = './dataset/augment_dataset/'
# 增强的图像的保存目录，此处好像只能用绝对路径==，算是一个小瑕疵 
AUGMENT_OUTPUT_DIR = 'F:/Pycharm Projects/Unet/dataset/augment_output'
 
def augment():
    if not os.path.exists(AUGMENT_OUTPUT_DIR):
        os.mkdir(AUGMENT_OUTPUT_DIR)
    # 获取每一张图像的路径
    filenames = glob.glob(os.path.join(AUGMENT_SOURCE_DIR, '*.png'))
    for filename in filenames:
        # 这里source_directory是单张图片，如果不需要同时生成标签，则这里直接填目录就好
        p = Augmentor.Pipeline(
            source_directory=filename,
            output_directory=AUGMENT_OUTPUT_DIR
        )
        # 图片对应的标签的目录，且二者必须同名（要自己预处理一下）
        p.ground_truth(ground_truth_directory=AUGMENT_LABEL_SOURCE_DIR)
        # 旋转：概率0.2
        p.rotate(probability=0.2, max_left_rotation=2, max_right_rotation=2)
        # 缩放
        p.zoom(probability=0.2, min_factor=1.1, max_factor=1.2)
        # 歪斜
        p.skew(probability=0.2)
        # 扭曲，注意grid_width, grid_height 不能超过原图
        p.random_distortion(probability=0.2, grid_width=20, grid_height=20, magnitude=1)
        # 四周裁剪
        p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
        # 随机裁剪
        p.crop_random(probability=0.2, percentage_area=0.8)
        # 翻转
        p.flip_random(probability=0.2)
        # 每张图片生成多少增强的图片
        p.sample(n=5)
 
augment()


def write_image_to_tfrecords():
    # image / label 各自的存储文件夹
    augment_image_path = AUGMENT_IMAGE_PATH
    augment_label_path = AUGMENT_LABEL_PATH
    # 要生成的文件：train、validation、predict
    train_set_writer = tf.python_io.TFRecordWriter(os.path.join('./dataset/my_set', TRAIN_SET_NAME))
    validation_set_writer = tf.python_io.TFRecordWriter(os.path.join('./dataset/my_set', VALIDATION_SET_NAME))
    predict_set_writer = tf.python_io.TFRecordWriter(os.path.join('./dataset/my_set', PREDICT_SET_NAME))
 
    # train set
    for idx in range(TRAIN_SET_SIZE):
        train_image = cv2.imread(os.path.join(augment_image_path, '%d.png' % idx))
        train_label = cv2.imread(os.path.join(augment_label_path, '%d.png' % idx), 0)
        train_image = cv2.resize(train_image, (INPUT_WIDTH, INPUT_HEIGHT))
        train_label = cv2.resize(train_label, (INPUT_WIDTH, INPUT_HEIGHT))
        train_label[train_label != 0] = 1
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_label.tobytes()])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_image.tobytes()]))
        }))     # example对象对label和image数据进行封装
        train_set_writer.write(example.SerializeToString())
        if idx % 100 == 0:
            print('Done train_set writing %.2f%%' % (idx / TRAIN_SET_SIZE * 100))
    train_set_writer.close()
    print('Done test set writing.')
 
    # validation set
    for idx in range(TRAIN_SET_SIZE, TRAIN_SET_SIZE + VALIDATION_SET_SIZE):
        validation_image = cv2.imread(os.path.join(augment_image_path, '%d.png' % idx))
        validation_label = cv2.imread(os.path.join(augment_label_path, '%d.png' % idx), 0)
        validation_image = cv2.resize(validation_image, (INPUT_WIDTH, INPUT_HEIGHT))
        validation_label = cv2.resize(validation_label, (INPUT_WIDTH, INPUT_HEIGHT))
        validation_label[validation_label != 0] = 1
 
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[validation_label.tobytes()])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[validation_image.tobytes()]))
        }))
        validation_set_writer.write(example.SerializeToString())  # 序列化为字符串
        if idx % 10 == 0:
            print('Done validation_set writing %.2f%%' % ((idx - TRAIN_SET_SIZE) / VALIDATION_SET_SIZE * 100))
    validation_set_writer.close()
    print("Done validation_set writing")
 
    # predict set
    predict_image_path = ORIGIN_PREDICT_IMG_DIR
    predict_label_path = ORIGIN_PREDICT_LBL_DIR
    for idx in range(PREDICT_SET_SIZE):
        predict_image = cv2.imread(os.path.join(predict_image_path, '%d.png'%idx))
        predict_label = cv2.imread(os.path.join(predict_label_path, '%d.png'%idx), 0)
        predict_image = cv2.resize(predict_image, (INPUT_WIDTH, INPUT_HEIGHT))
        predict_label = cv2.resize(predict_label, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        predict_label[predict_label != 0] = 1
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_label.tobytes()])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_image.tobytes()]))
        }))
        predict_set_writer.write(example.SerializeToString())
        if idx % 10 == 0:
            print('Done predict_set writing %.2f%%' % (idx / PREDICT_SET_SIZE * 100))
    predict_set_writer.close()
    print("Done predict_set writing")
