# -*- coding: utf-8 -*-

import os, shutil
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential 


# model = models.Sequential()
# model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(240,320,3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))


# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])


# model.summary()

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


def model_predict(model, class_index): # label
    files_name=[]
    predics=[]
    labels=[]

    for i in range(len(class_index)):
        # predict에 들어가는 데이터 개수 확인
        if i % 500 == 0:
          print(i, end=', ')
        try:
            test_path = "/home/juno/workspace/codetrain/trying_data/{}".format(class_index[i][:12])
            img = cv2.imread(test_path)

        except Exception as e:
            print(str(e))
        x = img.copy()
        x = img.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        pred = model.predict(x)
        files_name.append(class_index[i])
        predics.append(pred)
        # labels.append(label)

    re_dic = dict({'files':files_name, 'pred':predics}) # , 'labels':labels

    return re_dic
       
        # test_image = Image.open(test_path).resize((240,320))
        # img_tensor = np.array(test_image)
        # img_expand = np.expand_dims(img_tensor, axis=1)
        # test_data = img_expand / 255.0

        
# test_image = Image.open(test_path).resize((240, 320))
# img_tensor = np.array(test_image)
# img_tensor.shape
# img_expand = np.expand_dims(img_tensor, axis=0)
# x = img_expand / 255.0

def grad_cam(model, class_predict):
    plt.figure(figsize=(32, 32))

    for i in range(len(class_predict['pred'])): 
        test_path = os.path.join('/home/juno/workspace/codetrain/trying_data/{}'.format(class_predict['files'][i]))

        
        image = cv2.imread(test_path)
        img = cv2.resize(image, (320, 240))
        img = image.copy()
        x = img.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer('conv2d_2').output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(x, tf.float32) # 실수 자료형 변형
            model_outputs, predictions = grad_model(inputs) # model_outputs = 마지막 레이어 output, predictions = model.output = (none, 1)
            class_channel = predictions[:,0] # [:,0] 모든행에 대해 첫번째 열의 정보를 가져오기

        grads = tape.gradient(class_channel, model_outputs) # 마지막 컨볼루젼 출력 맵에 대한 기울기


        # shape [0,2] ... ? 해결하기
        prediction = predictions[0]
        model_outputs = model_outputs[0]

        # plt.subplot((len(class_predict['pred']) / 3) + 1, 3, i + 1)
        plt.subplots(figsize=(7, 7))
        
        plt.suptitle('{} Grad CAM'.format(class_predict['files'][0][:3]))
        plt.title('%s, pred: %0.4f'%(class_predict['files'][i], class_predict['pred'][i]))

        weights = np.mean(grads, axis=(1, 2)) # 가중치 평균  #grads = [[8, 5, 4, 000] , [5, 3, 1, 0, 5] , [3, 1, 0.1] -> array([[1, 3, 5, 3, 1, 2]]) 1차원 array
        weights = weights.reshape(16, 1) # array(4,8) -> (32,1)

        cam = (prediction -0.5) * np.matmul(model_outputs, weights) # matmul(3차원, )
        cam = np.mean(cam, axis=2)
        # cam 2차원 전환 시 숫자 조정
        cam -= np.min(cam)
        cam /= np.max(cam)


        try:
          cam = cv2.resize(np.float32(255*cam), (320, 240))
        except Exception as e: 
            print(str(e))

       
        #heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(np.uint8(cam), cv2.CV_8UC1)
        heatmap[np.where(cam <= 0.4)] = 0
        grad_cam = cv2.addWeighted(image, 0.8, heatmap, 0.4, 0)
        plt.axis('off')
        plt.imshow(grad_cam[:, :, ::-1])
        plt.savefig('/home/juno/workspace/codetrain/trainset/model_test/fail/{}'.format(class_predict['files'][i]))
        plt.close()



# 샛별선임님
# tensor (240, 320, 2)
# max, min 분류 good
get_model = tf.keras.models.load_model('/home/juno/workspace/codetrain/trying_model/lungsound_final_model.h5')

# 준호 (240, 320)
# epoch 40 = 과적합 발생 전
# max, min 동일
# get_model = tf.keras.models.load_model('/home/juno/workspace/codetrain/test_binary.h5')


# 파일접근 경로 설정
# 파일 접근
lung_path = '/home/juno/workspace/codetrain/trying_data'
# test_path = '/home/juno/workspace/codetrain/test1234'

## 폴더별 접근
all_files = os.listdir(lung_path)
# fail_path = os.path.join(test_path, os.listdir(test_path)[2])
# none_path = os.path.join(test_path, os.listdir(test_path)[0])

# fail_files = os.listdir(fail_path)
# none_files = os.listdir(none_path)

# predict_call
all_predict = model_predict(get_model, all_files)
# none_predict = model_predict(get_model, none_files)
# fail_predict = model_predict(get_model, fail_files)

# predict_check
for i in range(15):
    print(all_predict['files'][i], all_predict['pred'][i][0][0], all_predict['pred'][i][0][1]) #, fail_predict['labels'][i]

# none
file_0_name = []
file_0_pred = []
# fail
file_1_name = []
file_1_pred = []

# none/fail_mean
# none_mean_c = []
# fail_mean_c = []


# for i in range(len(all_predict['pred'])): # none > fail
#     if all_predict['pred'][i][0][1] > 0.600:
#         file_0_name.append(all_predict['files'][i])
#         file_0_pred.append(all_predict['pred'][i][0][1]) # array([[]])
#     else:
#         file_1_name.append(all_predict['files'][i])
#         file_1_pred.append(all_predict['pred'][i][0][1])      

# none_result = dict({'files':file_0_name, 'pred': file_0_pred})        
# fail_result = dict({'files':file_1_name, 'pred': file_1_pred})


# grad_cam(get_model, none_result)


# none_mean 구하기
# for i in range(len(none_predict['pred'])): 
#     # if none_predict['pred'][i]:
#     none_mean = (none_predict['pred'][i][0][1])
#     none_mean_c.append(none_mean)

# none_mean_c = np.mean(fail_mean_c)

# fail_mean 구하기

# for i in range(len(fail_predict)):
#     print(fail_predict['pred'][i][0][0] + " // " + fail_predict['pred'][i][0][1])

# for i in range(len(fail_predict['pred'])): 
#     fail_mean = (fail_predict['pred'][i][0])
#     fail_mean_c.append(fail_mean)

# fail_mean_c = np.mean(fail_mean_c)


# for i in range(len(all_predict['pred'])): # none > fail
#     if none_mean_c > fail_mean_c:    ##  
#         file_0_name.append(all_predict['files'][i])
#         file_0_pred.append(all_predict['pred'][i][0][1]) # array([[]]) # (pred array([[min, max ]]) not (max, min))
#     else:
#         file_1_name.append(all_predict['files'][i])
#         file_1_pred.append(all_predict['pred'][i][0][1])    


for i in range(len(all_predict['pred'])): # none > fail
    if all_predict['pred'][i][0][0] > all_predict['pred'][i][0][1]:
        file_0_name.append(all_predict['files'][i])
        file_0_pred.append(all_predict['pred'][i][0][0]) # array([[]]) # (pred array([[none %, fail% ]]) not (max, min))
    else:
        file_1_name.append(all_predict['files'][i])
        file_1_pred.append(all_predict['pred'][i][0][1])        


none_result = dict({'files':file_0_name, 'pred': file_0_pred})        
fail_result = dict({'files':file_1_name, 'pred': file_1_pred})


# grad_CAM 콜
# none 구분
grad_cam(get_model, none_result)

# fail 구분
# grad_cam(get_model, fail_result)


# predict_classification
# fail_files 중 일치/불일치 구분
# fail_1 = 일치, fail_0 = 불일치
# fail_0_name = []
# fail_0_pred = []
# fail_1_name = []
# fail_1_pred = []

# for i in range(len(fail_predict['pred'])):
#     if fail_predict['pred'][i][0][0] > 0.980:
#         fail_1_name.append(fail_predict['files'][i])
#         fail_1_pred.append(fail_predict['pred'][i][0][0])
#     else:
#         fail_0_name.append(fail_predict['files'][i])
#         fail_0_pred.append(fail_predict['pred'][i][0][0])

# fail_0 = dict({'files':fail_0_name, 'pred':fail_0_pred})        
# fail_1 = dict({'files':fail_1_name, 'pred':fail_1_pred})





# grad_cam(get_model, fail_0)


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


'''
for i in range(2):
    string += format(predictions[0][i], "10.2f") + ' ' # prodictions[0[i]]
            
    if predictions[0][0] > predictions[0][1]:
        #print(file)
        print(string + '           -> normal')
        #n_cnt += 1
    else:
        print(string + '           -> abnormal')
        #ab_cnt += 1
'''