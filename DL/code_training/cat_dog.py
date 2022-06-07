# -*- coding: utf-8 -*-
import os, shutil

# # 원본 데이터셋을 압축 해제한 디렉터리 경로
# original_dataset_dir = './datasets/cats_and_dogs/train'

# # 소규모 데이터셋을 저장할 디렉터리
# base_dir = './datasets/cats_and_dogs_small'
# if os.path.exists(base_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
#     shutil.rmtree(base_dir)   # 이 코드는 책에 포함되어 있지 않습니다.
# os.mkdir(base_dir)

# # 훈련, 검증, 테스트 분할을 위한 디렉터리
# train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
# validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

# # 훈련용 고양이 사진 디렉터리
# train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)

# # 훈련용 강아지 사진 디렉터리
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)

# # 검증용 고양이 사진 디렉터리
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)

# # 검증용 강아지 사진 디렉터리
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)

# # 테스트용 고양이 사진 디렉터리
# test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)

# # 테스트용 강아지 사진 디렉터리
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)

# # 처음 1,000개의 고양이 이미지를 train_cats_dir에 복사합니다
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)

# # 다음 500개 고양이 이미지를 validation_cats_dir에 복사합니다
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
    
# # 다음 500개 고양이 이미지를 test_cats_dir에 복사합니다
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)
    
# # 처음 1,000개의 강아지 이미지를 train_dogs_dir에 복사합니다
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
    
# # 다음 500개 강아지 이미지를 validation_dogs_dir에 복사합니다
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
    
# # 다음 500개 강아지 이미지를 test_dogs_dir에 복사합니다
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)

from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential 


model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
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
# train_dir = '/home/juno/workspace/codetrain/cats_and_dogs2/train'
# validation_dir = '/home/juno/workspace/codetrain/cats_and_dogs2/validation'



# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # 모든 이미지를 1/255로 스케일을 조정합니다
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         # 타깃 디렉터리
#         train_dir,
#         # 모든 이미지를 150 × 150 크기로 바꿉니다
#         target_size=(150, 150),
#         batch_size=128,
#         # binary_crossentropy 손실을 사용하기 때문에 이진 레이블이 필요합니다
#         class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=128,
#         class_mode='binary')

# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=15,
#       epochs=15,
#       validation_data=validation_generator,
#       validation_steps=15)

# model.save('cats_and_dogs_15epoch_3layers.h5')

import tensorflow as tf
from tensorflow.keras.preprocessing import image

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pickle

from PIL import Image

def model_predict(model, class_index, label):
    files_name=[]
    predics=[]
    labels=[]

    for i in range(len(class_index)):
        if i % 100 == 0:
          print(i, end=', ')
        test_path = '/home/juno/workspace/codetrain/cats_and_dogs2/test/{}/{}'.format(class_index[i][:3], class_index[i])
        test_image = Image.open(test_path).resize((150, 150))

        img_tensor = np.array(test_image)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        test_data = img_tensor / 255.0

        pred = model.predict(test_data)

        files_name.append(class_index[i])
        predics.append(pred)
        labels.append(label)

    re_dic = dict({'files':files_name, 'pred':predics, 'labels':labels})

    return re_dic



def grad_cam(model, class_predict):
    plt.figure(figsize=(32, 32))

    for i in range(len(class_predict['pred'])): 
        path = os.path.join('/home/juno/workspace/codetrain/cats_and_dogs2/test/{}/{}'.format(class_predict['files'][0][:3], class_predict['files'][i]))
        image = cv2.imread(path)
        img = cv2.resize(image, (150, 150))
        x = img.copy()
        x.astype(np.float32)
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
        weights = weights.reshape(32, 1)

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


get_model = tf.keras.models.load_model('/home/juno/workspace/codetrain/cats_and_dogs_100epoch_3layers.h5')

test_path = '/home/juno/workspace/codetrain/cats_and_dogs2/test'

cat_path = os.path.join(test_path, os.listdir(test_path)[0])
dog_path = os.path.join(test_path, os.listdir(test_path)[1])

cat_files = os.listdir(cat_path)
dog_files = os.listdir(dog_path)

# 고양이 예측갑 구하기

cats_predict = model_predict(get_model, cat_files, 0)

for i in range(10):
    print(cats_predict['files'][i], cats_predict['pred'][i][0][0], cats_predict['labels'][i])


# 고양이 분류
cats_0_name = []
cats_0_pred = []
cats_1_name = []
cats_1_pred = []

for i in range(len(cats_predict['pred'])):
    if cats_predict['pred'][i][0][0] > 0.500:
        cats_1_name.append(cats_predict['files'][i])
        cats_1_pred.append(cats_predict['pred'][i][0][0])
    else:
        cats_0_name.append(cats_predict['files'][i])
        cats_0_pred.append(cats_predict['pred'][i][0][0])

cat_0 = dict({'files':cats_0_name, 'pred':cats_0_pred})        
cat_1 = dict({'files':cats_1_name, 'pred':cats_1_pred})

cat_1

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

cat_1

grad_cam(get_model, cat_1)