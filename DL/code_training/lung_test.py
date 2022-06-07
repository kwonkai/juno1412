# -*- coding: utf-8 -*-

import os, shutil
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential 


model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(240,320,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


model.summary()

# 훈련, 검증, 테스트 분할을 위한 디렉터리
# train_dir = '/home/juno/workspace/codetrain/trainset/train'
# validation_dir = '/home/juno/workspace/codetrain/trainset/val'

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모든 이미지를 1/255로 스케일을 조정합니다
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         # 타깃 디렉터리
#         train_dir,
#         # 모든 이미지를 150 × 150 크기로 바꿉니다
#         target_size=(240, 320),
#         batch_size=128,
#         # binary_crossentropy 손실을 사용하기 때문에 이진 레이블이 필요합니다
#         class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(240, 320),
#         batch_size=128,
#         class_mode='binary')

# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=12,
#       epochs=30,
#       validation_data=validation_generator,
#       validation_steps=5)

# model.save('lung_sound_test.h5')

import tensorflow as tf
from tensorflow.keras.preprocessing import image

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from PIL import Image

# test = '/home/juno/workspace/codetrain/trainset/test/fai/fail666.png'
# test_image = Image.open(test)
# Image.show()
# test_image
# test_image = np.array(test)
# img_tensor = np.expand_dims(test_image, axis=0)


def model_predict(model, class_index, label):
    files_name=[]
    predics=[]
    labels=[]

    for i in range(len(class_index)):
        if i % 100 == 0:
          print(i, end=', ')
        try:
            test_path = "/home/juno/workspace/codetrain/trainset/test/{}/{}".format(class_index[i][:4], class_index[i])
            image = cv2.imread(test_path)
            img = cv2.resize(image, (240, 320))

        except Exception as e:
            print(str(e))
        x = img.copy()
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        pred = model.predict(x)
        files_name.append(class_index[i])
        predics.append(pred)
        labels.append(label)

    re_dic = dict({'files':files_name, 'pred':predics, 'labels':labels})

    return re_dic
       
        # test_image = Image.open(test_path).resize((240,320))
        # img_tensor = np.array(test_image)
        # img_expand = np.expand_dims(img_tensor, axis=1)
        # test_data = img_expand / 255.0

        


def grad_cam(model, class_predict):
    plt.figure(figsize=(32, 32))

    for i in range(len(class_predict['pred'])): 
        test_path = os.path.join('/home/juno/workspace/codetrain/trainset/test/{}/{}'.format(class_predict['files'][0][:4], class_predict['files'][i]))
        image = cv2.imread(test_path)
        img = cv2.resize(image, (240, 320))
        x = img.copy()
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer('conv2d_2').output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(x, tf.float32)
            model_outputs, predictions = grad_model(inputs)
            loss = predictions[:,0]


        grads = tape.gradient(loss, model_outputs)

        guided_grads = (
            tf.cast(model_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
        )



        prediction = predictions[0]
        model_outputs = model_outputs[0]

        plt.subplot((len(class_predict['pred']) / 3) + 1, 3, i + 1)
        plt.suptitle('{} Grad CAM'.format(class_predict['files'][0][:3]))
        plt.title('%s, pred: %0.4f'%(class_predict['files'][i], class_predict['pred'][i]))

        weights = np.mean(grads, axis=(1, 2))
        weights = weights.reshape(64, 1)

        cam = (prediction -0.5) * np.matmul(model_outputs, weights)
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam -= 0.2
        cam /= 0.8

        try:
            cam = cv2.resize(np.float32(cam), (150, 150))
        except Exception as e: 
            print(str(e))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.2)] = 0
        grad_cam = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
        plt.axis('off')
        plt.imshow(grad_cam[:, :, ::-1])



get_model = tf.keras.models.load_model('/home/juno/workspace/codetrain/lung_sound_test.h5')

test_path = '/home/juno/workspace/codetrain/trainset/test'

fail_path = os.path.join(test_path, os.listdir(test_path)[0])
none_path = os.path.join(test_path, os.listdir(test_path)[1])

fail_files = os.listdir(fail_path)
none_files = os.listdir(none_path)

fail_predict = model_predict(get_model, fail_files, 0)

for i in range(5):
    print(fail_predict['files'][i], fail_predict['pred'][i][0][0], fail_predict['labels'][i])

fail_0_name = []
fail_0_pred = []
fail_1_name = []
fail_1_pred = []

for i in range(len(fail_predict['pred'])):
    if fail_predict['pred'][i][0][0] > 0.500:
        fail_1_name.append(fail_predict['files'][i])
        fail_1_pred.append(fail_predict['pred'][i][0][0])
    else:
        fail_0_name.append(fail_predict['files'][i])
        fail_0_pred.append(fail_predict['pred'][i][0][0])

fail_0 = dict({'files':fail_0_name, 'pred':fail_0_pred})        
fail_1 = dict({'files':fail_1_name, 'pred':fail_1_pred})

fail_1

grad_cam(get_model, fail_1)


# print('맞은 것')
# for i in range(10):
#     print(cat_0['files'][i], cat_0['pred'][i])
# print('틀린 것')
# for i in range(10):  
#     print(cat_1['files'][i], cat_0['pred'][i])

# 맞는 것
# with open('/home/juno/workspace/codetrain/cats_and_dogs2/cat_0.bin', 'wb') as hi:
#   pickle.dump(cat_0, hi)

# # 틀린 것
# with open('/home/juno/workspace/codetrain/cats_and_dogs2/cat_1.bin', 'wb') as hi:
#   pickle.dump(cat_1, hi)

