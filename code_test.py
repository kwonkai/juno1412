# 코드 트레이닝 폐소리 데이터 구분하기
# import PIL
# from PIL import Image 
# import cv2


import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

## 3.데이터 train/test/valudation set 분류
import os, shutil
import natsort

# from tensorflow.python.client import device_lib
# print(tf.__version__)
# print(device_lib.list_local_devices())


# 1. 모든데이터
# path 경로 
data_path = '/home/juno/workspace/codetrain/dataset' # 폴더경로
none_path = '/home/juno/workspace/codetrain/none'
none_path2 = '/home/juno/workspace/codetrain/png_adc_team_none'
fail_path = '/home/juno/workspace/codetrain/wheeze'
fail_path2 = '/home/juno/workspace/codetrain/both'
train_path = '/home/juno/workspace/codetrain/trainset'





# # 이미지 불러오기
# # trainset 폴더 제작(데이터 모두 통합 none1+none2+fail)
# os.mkdir(train_path)
# os.chdir(none_path) # 폴더로 이동
# none_files = os.listdir(none_path) # 해당 폴더에 있는 파일 이름 리스트 받기

# # 1-1. 파일명 변경하기
# # none1
# # 리스트 1, 2, 3, ~ 변경 # 1, 10, 11
# os.chdir(none_path) # 폴더로 이동
# none_files = os.listdir(none_path) # 해당 폴더에 있는 파일 이름 리스트 받기
# none_files = natsort.natsorted(none_files, reverse=False)

# i=1
# for none_file in none_files:
#     src = os.path.join(none_path, none_file)
#     dst = 'none'+ str(i) + '.png'
#     dst = os.path.join(none_path, dst)
#     os.rename(src, dst)
#     i+=1

# pics = ['none{}.png'.format(i) for i in range(1,2855)]
# for pic in pics:
#     src = os.path.join(none_path, pic)
#     dst = os.path.join(train_path, pic)
#     shutil.copy(src,dst)

# # none_path2

# os.chdir(none_path2) # 폴더로 이동
# none_files2 = os.listdir(none_path2) # 해당 폴더에 있는 파일 이름 리스트 받기
# none_files2 = natsort.natsorted(none_files2, reverse=False)

# i=2854
# for none_file in none_files2:
#     src = os.path.join(none_path2, none_file)
#     dst = 'none'+ str(i) + '.png'
#     dst = os.path.join(none_path2, dst)
#     os.rename(src, dst)
#     i+=1


# pics = ['none{}.png'.format(i+2854) for i in range(815)]
# for pic in pics:
#     src = os.path.join(none_path2, pic)
#     dst = os.path.join(train_path, pic)
#     shutil.copy(src,dst)


# # fail_path
# os.chdir(fail_path) # 폴더로 이동
# fail_files = os.listdir(fail_path)
# fail_files = natsort.natsorted(fail_files, reverse=False) # 해당 폴더에 있는 파일 이름 리스트 받기

# i=1
# for fail_file in fail_files:
#     src = os.path.join(fail_path, fail_file)
#     dst = 'fail'+ str(i) + '.png'
#     dst = os.path.join(fail_path, dst)
#     os.rename(src, dst)
#     i+=1

# pics = ['fail{}.png'.format(i) for i in range(1,515)]
# for pic in pics:
#     src = os.path.join(fail_path, pic)
#     dst = os.path.join(train_path, pic)
#     shutil.copy(src,dst)

# # fail_path2
# os.chdir(fail_path2) # 폴더로 이동
# fail_files2 = os.listdir(fail_path2)
# fail_files2 = natsort.natsorted(fail_files2, reverse=False) # 해당 폴더에 있는 파일 이름 리스트 받기

# i=515
# for fail_file2 in fail_files2:
#     src = os.path.join(fail_path2, fail_file2)
#     dst = 'fail'+ str(i) + '.png'
#     dst = os.path.join(fail_path2, dst)
#     os.rename(src, dst)
#     i+=1

# pics = ['fail{}.png'.format(i+515) for i in range(292)]
# for pic in pics:
#     src = os.path.join(fail_path2, pic)
#     dst = os.path.join(train_path, pic)
#     shutil.copy(src,dst)

# print("This folder has already been created")



# # 1-2훈련/검증/테스트 분할 폴더 생성 + 파일 복사
# # 함수화, 줄여야함...

# data_path ='/home/juno/workspace/codetrain/trainset'
# train_pic = os.path.join(data_path, 'train')
# os.mkdir(train_pic)
# val_pic = os.path.join(data_path, 'val')
# os.mkdir(val_pic)
# test_pic = os.path.join(data_path, 'test')
# os.mkdir(test_pic)

# # train/validation/test 디렉터리의 none/fail data
# train_none_dir = os.path.join(train_pic, 'none')
# os.mkdir(train_none_dir)

# train_fail_dir = os.path.join(train_pic, 'fail')
# os.mkdir(train_fail_dir)

# # validation
# val_none_dir = os.path.join(val_pic, 'none')
# os.mkdir(val_none_dir)

# val_fail_dir = os.path.join(val_pic, 'fail')
# os.mkdir(val_fail_dir)

# # test
# test_none_dir = os.path.join(test_pic, 'none')
# os.mkdir(test_none_dir)

# test_fail_dir = os.path.join(test_pic, 'fail')
# os.mkdir(test_fail_dir)


# # 파일 복사하기
# # 위 생성한 데이터 폴더에 none/fail 데이터 복사하기
# # none 데이터
# pics = ['none{}.png'.format(i) for i in range(1,2400)]
# for pic in pics:
#     src = os.path.join(none_path, pic)
#     dst = os.path.join(train_none_dir, pic)
#     shutil.copy(src,dst)

# pics = ['none{}.png'.format(i) for i in range(2401, 3000)]
# for pic in pics:
#     src = os.path.join(none_path, pic)
#     dst = os.path.join(val_none_dir, pic)
#     shutil.copy(src,dst)

# pics = ['none{}.png'.format(i) for i in range(3001, 3668)]
# for pic in pics:
#     src = os.path.join(none_path, pic)
#     dst = os.path.join(test_none_dir, pic)
#     shutil.copy(src,dst)


# # fail 데이터
# pics = ['fail{}.png'.format(i) for i in range(1,480)]
# for pic in pics:
#     src = os.path.join(fail_path, pic)
#     dst = os.path.join(train_fail_dir, pic)
#     shutil.copy(src,dst)

# pics = ['fail{}.png'.format(i) for i in range(481, 515)]
# for pic in pics:
#     src = os.path.join(fail_path, pic)
#     dst = os.path.join(val_fail_dir, pic)
#     shutil.copy(src,dst)
# pics = ['fail{}.png'.format(i) for i in range(516, 640)]
# for pic in pics:
#     src = os.path.join(fail_path2, pic)
#     dst = os.path.join(val_fail_dir, pic)
#     shutil.copy(src,dst)


# pics = ['fail{}.png'.format(i) for i in range(641, 807)]
# for pic in pics:
#     src = os.path.join(fail_path2, pic)
#     dst = os.path.join(test_fail_dir, pic)
#     shutil.copy(src,dst)


# print("Data has already been copied")

# cnn modeling

# model = models.Sequential()

# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(240,320,3)))
# model.add(layers.MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(layers.Conv2D(64,(3,3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(layers.Conv2D(128,(3,3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(layers.Flatten())
# model.add(Dropout(0.3))

# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 선임님 cnn model
model = models.Sequential()
model.add(layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(240,320,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(16, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (1,1), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

from tensorflow.keras import optimizers
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr=1e-4), metrics=['accuracy'])






# trainset 폴더 경로
train_pic = ('/home/juno/workspace/codetrain/trainset/train')
test_pic = ('/home/juno/workspace/codetrain/trainset/test')
val_pic = ('/home/juno/workspace/codetrain/trainset/val')

## generator
# def generator_train():
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_pic, # 폴더 지정
        target_size=(240,320),
        batch_size=256,
        class_mode='binary'
)
val_generator = test_datagen.flow_from_directory(
        val_pic, # 폴더 지정
        target_size=(240,320),
        batch_size=256,
        class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
        test_pic, # 폴더 지정
        target_size=(240,320),
        batch_size=256,
        class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기:', data_batch.shape)
    print('배치 레이블 크기:', labels_batch.shape)
    break

# model update
model_dir = '/home/juno/workspace/codetrain/model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

modelpath = '/home/juno/workspace/codetrain/model/test{epoch:04d}-{loss:.4f}.hdf5'

# model checkpoint : earlystop, checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpointer = ModelCheckpoint(filepath = modelpath, monitor='val_loss', verbose = 1, save_best_only=True )
early_stop = EarlyStopping(monitor='val_loss', patience=200)

## generator model train
# model = cnn()
# train_generator, val_generator = generator_train()

# logits and labels must have the same shape ((None, 2) vs (None, 1))

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 12,
    epochs = 2000,
    validation_data = val_generator,
    validation_steps = 3,
    callbacks = [early_stop, checkpointer],
)


# # model save
model.save('best_model100.hdf5')
# lung_model = load_model('best_model3.hdf5')


# history save
np.save('best_model_hist100.npy', history.history)
# history = np.load('best_model_his3.npy', allow_pickle='True').item()


# history save
# import json
# with open('best_model_hist.json', 'w') as f:
#     json.dump(history.history, f)




# pickle
# import pickle
# with open('/best_model_hist', 'wb') as file_fi:
#     pickle.dump(history.history, file_fi)



# 결과 시각화 - 복구하기
# matplotlib 안먹힘...
# acc = history.history['accuracy']
# loss = history.history['loss']
# # loss = len(loss)
# val_acc = history.history['val_accuracy']
# # val_acc = len(val_acc)
# val_loss = history.history['val_loss']
# # val_loss = len(val_loss)


# epochs = range(1, len(acc) + 1)


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()





# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'bo', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.show()

# # plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()








        
# from keras.preprocessing.text import text_to_word_sequence
# text_to_word_sequence(none_files[0])

# # 2-2. label 변환
# label = []
# for n, path in enumerate(none_files[:100]):
#     token = text_to_word_sequence[none_files[n]]
#     label.append[token(0)]