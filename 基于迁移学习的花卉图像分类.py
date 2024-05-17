import os, glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 变量
resize = 224  # 图片尺寸参数
epochs = 8  # 迭代次数
batch_size = 5  # 每次训练多少张

# ——————————————————————————————————————————————————————————————————————————————————

# 训练集路径
train_data_path = "D:\大学课程\人工智能\Training"
# 玫瑰花文件夹路径
daisy_path = os.path.join(train_data_path, 'daisy')
# 太阳花文件夹路径
tulip_path = os.path.join(train_data_path, 'tulip')

# 将文件夹内的图片读取出来
fpath_daisy = [os.path.abspath(fp) for fp in glob.glob(os.path.join(daisy_path, '*.jpg'))]
fpath_tulip = [os.path.abspath(fp) for fp in glob.glob(os.path.join(tulip_path, '*.jpg'))]

# 文件数量
num_daisy = len(fpath_daisy)
num_tulip = len(fpath_tulip)

# 设置标签
label_daisy = [0] * num_daisy
label_tulip = [1] * num_tulip

# 展示
print('daisy:   ', num_daisy)
print('tulip: ', num_tulip)

# 划分为多少验证集
RATIO_TEST = 0.1

num_daisy_test = int(num_daisy * RATIO_TEST)
num_tulip_test = int(num_tulip * RATIO_TEST)

# train
fpath_train = fpath_daisy[num_daisy_test:] + fpath_tulip[num_tulip_test:]
label_train = label_daisy[num_daisy_test:] + label_tulip[num_tulip_test:]

# validation
fpath_vali = fpath_daisy[:num_daisy_test] + fpath_tulip[:num_tulip_test]
label_vali = label_daisy[:num_daisy_test] + label_tulip[:num_tulip_test]

num_train = len(fpath_train)
num_vali = len(fpath_vali)

# 展示
print('num_train:   ', num_train)
print('num_label: ', num_vali)


# 预处理函数
def preproc(fpath, label):
    image_byte = tf.io.read_file(fpath)  # 读取文件
    image = tf.io.decode_image(image_byte)  # 检测图像是否为BMP,GIF,JPEG或PNG,并执行相应的操作将输入字节string转换为类型uint8的Tensor
    image_resize = tf.image.resize_with_pad(image, 224, 224)  # 缩放到224*224
    image_norm = tf.cast(image_resize, tf.float32) / 255.  # 归一化

    label_onehot = tf.one_hot(label, 2)

    return image_norm, label_onehot


dataset_train = tf.data.Dataset.from_tensor_slices((fpath_train, label_train))  # 将数据进行预处理
dataset_train = dataset_train.shuffle(num_train).repeat()  # 打乱顺序
dataset_train = dataset_train.map(preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_train = dataset_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)  # 一批次处理多少份

dataset_vali = tf.data.Dataset.from_tensor_slices((fpath_vali, label_vali))
dataset_vali = dataset_vali.shuffle(num_vali).repeat()
dataset_vali = dataset_vali.map(preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_vali = dataset_vali.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# ——————————————————————————————————————————————————————————————————————————————————

# 建立模型 卷积神经网络
model = tf.keras.Sequential(name='Alexnet')
# 第一层
model.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                        strides=(4, 4), padding='valid',
                        input_shape=(resize, resize, 3),
                        activation='relu'))
model.add(layers.BatchNormalization())
# 第一层池化层：最大池化层
model.add(layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='valid'))

# 第二层
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                        strides=(1, 1), padding='same',
                        activation='relu'))
model.add(layers.BatchNormalization())
# 第二层池化层
model.add(layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='valid'))

# 第三层
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), padding='same',
                        activation='relu'))
# 第四层
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), padding='same',
                        activation='relu'))
# 第五层
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                        strides=(1, 1), padding='same',
                        activation='relu'))
# 池化层
model.add(layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2), padding='valid'))

# 第6，7，8层
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.5))

# Output Layer
model.add(layers.Dense(2, activation='softmax'))

# Training 优化器 随机梯度下降算法
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',  # 梯度下降法
              metrics=['accuracy'])

history = model.fit(dataset_train,
                    steps_per_epoch=num_train // batch_size,
                    epochs=epochs,  # 迭代次数
                    validation_data=dataset_vali,
                    validation_steps=num_vali // batch_size,
                    verbose=1)

# 评分标准
scores_train = model.evaluate(dataset_train, steps=num_train // batch_size, verbose=1)
print(scores_train)

scores_vali = model.evaluate(dataset_vali, steps=num_vali // batch_size, verbose=1)
print(scores_vali)

# 保存模型
model.save('D:\大学课程\人工智能')

history_dict = history.history
train_loss = history_dict['loss']
train_accuracy = history_dict['accuracy']
val_loss = history_dict['val_loss']
val_accuracy = history_dict['val_accuracy']

# Draw loss
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# Draw accuracy
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')

# Display
plt.show()

print('Train has finished')