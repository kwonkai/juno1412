# 코드 트레이닝 폐소리 데이터 구분하기
from tkinter import Label
import pandas as pd
import numpy as np
import os
from PIL import Image 
from glob import glob
import cv2
import matplotlib.pyplot as plt
from string import digits
import shutil
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
'''
# 데이터 전처리 삽질1
# j=1
# for j in enumerate(none_files):
#     none_img = Image.open(path, none_files[j])
#     none_img = none_img.convert("RGB")
#     none_img = none_img.resize(img, (20,20))
#     none_img = np.asarray(img)
#     j += 1

# none_files = none_files.reshape(none_files.shape[0])

'''

# 1. 데이터 불러오기
# 1-1. both 데이터 불러오기

# 1-2. none1 image 불러오기
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/none' # 폴더경로
os.chdir(path) # 폴더로 이동
none_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기
len(none_files)

# none 사진파일 가져오기
png_none = []
for none_file in none_files:
    if '.png' in none_file:
        f = cv2.imread(none_file)
        png_none.append(f)

# 차원 축소
png = np.array(png_none)
png_none = png.reshape(-1, 320*240*3) # 3차원 -> 1차원 축소
png_none = png_none.reshape(-1, 320*240*3)/255 # 정규화
png_none

# 1-3. none1 image 불러오기
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/png_adc_team_none' # 폴더경로
os.chdir(path) # 폴더로 이동
none_files2 = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기
len(none_files2)

# none 사진파일 가져오기
png_none2 = []
for none_file in none_files2:
    if '.png' in none_file:
        f = cv2.imread(none_file)
        png_none2.append(f)

# 차원 축소
png2 = np.array(png_none2)
png_none2 = png2.reshape(-1, 320*240*3) # 3차원 -> 1차원 축소
png_none2 = png_none2.reshape(-1, 320*240*3)/255 # 정규화
png_none2


# 1-4, none 값 합하기
png_none_plus = np.block([[png_none], [png_none2]])
png_none_plus


# wheeze 데이터 - fail
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/wheeze' # 폴더경로
os.chdir(path) # 폴더로 이동
wheeze_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기
len(wheeze_files)

# none 사진파일 가져오기
png_wheeze = []
for wheeze_file in wheeze_files:
    if '.png' in wheeze_file:
        f = cv2.imread(wheeze_file)
        png_wheeze.append(f)

# 차원 축소
png = np.array(png_wheeze)
png_wheeze = png.reshape(-1, 240*320*3) # 3차원 -> 1차원 축소
png_wheeze = png_wheeze.reshape(-1, 240*320*3)/255 # 정규화
png_wheeze


### 2. 라벨링 - 수정...
# 정상데이터 none = 1
nonedatas = os.listdir('C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/traintest')
normals = []
for nonedata in nonedatas:
    normal = nonedata.split(',')[0]
    if normal == '_':
        normals.append(1)

df = pd.DataFrame({'image' : nonedatas, 'normal' : normals})
df

# 비정상데이터 wheeze = 1
re_nonedatas = os.listdir('C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/wheeze')
re_normals = []
for re_nonedata in re_nonedatas:
    re_normal = re_nonedata.split(',')[0]
    if re_normal == '_':
        re_normals.append(0)

df2 = pd.DataFrame({'image' : re_nonedatas, 'normal' : re_normals})
df2


# 



# '''
# # 폴더만들기
# ## 3.데이터 train/test/valudation set 분류
# import os, shutil
# from distutils.dir_util import copytree


# ### 데이터 폴더 합하기 해결할 거쇼

# # 1. 모든데이터
# # 1-1. 성공/실패 데이터 폴더 만들기
# root ='C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/traintest'
# os.mkdir(root)

# # none data 복사
# shutil.copytree('C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/none', 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/traintest')


# shutil.copy('C:\\Users\\kwonk\\juno1412-1\\juno1412\\DL\\code_training\\datasets_lungsound (2)\\wheeze', 'C:\\Users\\kwonk\\juno1412-1\\juno1412\\DL\\code_training\\datasets_lungsound (2)\\traintest')
# shutil.copy('C:\\Users\\kwonk\\juno1412-1\\juno1412\\DL\\code_training\\datasets_lungsound (2)\\png_abc_team_none', 'C:\\Users\\kwonk\\juno1412-1\\juno1412\\DL\\code_training\\datasets_lungsound (2)\\traintest')

# data_path = 'C:\\Users\\kwonk\\juno1412-1\\juno1412\\DL\\code_training\\datasets_lungsound\\traintest' # 폴더경로
# os.chdir(data_path) # 폴더로 이동
# none_files = os.listdir(data_path) # 해당 폴더에 있는 파일 이름 리스트 받기
# # none/fail 데이터 이동 코드 필요 # 현재 그냥 마우스로 옮김
# '''


# 훈련/검증/테스트 분할 폴더
# 훈련
data_path ='C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound (2)/dataset'
os.mkdir(data_path)

train_pic = os.path.join(data_path, 'train')
os.mkdir(train_pic)
val_pic = os.path.join(data_path, 'val')
os.mkdir(val_pic)
test_pic = os.path.join(data_path, 'test')
os.mkdir(test_pic)

# train/validation/test 디렉터리의 none/fail data
train_none_dir = os.path.join(train_pic, 'none')
os.mkdir(train_none_dir)

train_fail_dir = os.path.join(train_pic, 'fail')
os.mkdir(train_fail_dir)

# validation
val_none_dir = os.path.join(val_pic, 'none')
os.mkdir(val_none_dir)

val_fail_dir = os.path.join(val_pic, 'fail')
os.mkdir(val_fail_dir)

# test
test_none_dir = os.path.join(test_pic, 'none')
os.mkdir(test_none_dir)

test_fail_dir = os.path.join(test_pic, 'fail')
os.mkdir(test_fail_dir)


# 파일 복사하기
# 위 생성한 데이터 폴더에 none/fail 데이터 복사하기
# none 데이터
pics = ['none{}.png'.format(i) for i in range(1,1500)]
for pic in pics:
    src = os.path.join(data_path, pic)
    dst = os.path.join(train_none_dir, pic)
    shutil.copy(src,dst)

pics = ['none{}.png'.format(i) for i in range(1500, 2200)]
for pic in pics:
    src = os.path.join(data_path, pic)
    dst = os.path.join(val_none_dir, pic)
    shutil.copy(src,dst)

pics = ['none{}.png'.format(i) for i in range(2200, 2800)]
for pic in pics:
    src = os.path.join(data_path, pic)
    dst = os.path.join(test_none_dir, pic)
    shutil.copy(src,dst)


# fail 데이터
pics = ['none{}.png'.format(i) for i in range(1,300)]
for pic in pics:
    src = os.path.join(data_path, pic)
    dst = os.path.join(train_fail_dir, pic)
    shutil.copy(src,dst)

pics = ['none{}.png'.format(i) for i in range(300, 400)]
for pic in pics:
    src = os.path.join(data_path, pic)
    dst = os.path.join(val_fail_dir, pic)
    shutil.copy(src,dst)

pics = ['none{}.png'.format(i) for i in range(400, 515)]
for pic in pics:
    src = os.path.join(data_path, pic)
    dst = os.path.join(test_fail_dir, pic)
    shutil.copy(src,dst)


# 데이터 전처리
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        png_none, # 폴더 지정
        target_size=(80,60), # 사진 크기 지정
        batch_size=20,
        class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
        png_wheeze, # 폴더 지정
        target_size=(80,60), #사진크기지정
        batch_size=20,
        class_mode='binary'
)





# 라벨링
# none_files[:10]
# 2-1. 텍스트 토큰화
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(png_wheeze)

# 2-2. label 변환
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail'
label = []
for n, path in enumerate(none_files[:100]):
    token = text_to_word_sequence(none_files[n])
    label.append(token[0])

label

# 2-3. label Encoding
from sklearn.preprocessing import LabelEncoder
image = label
encoder = LabelEncoder()
encoder.fit(image)
label = encoder.transform(encoder)









# fail 데이터 정리
# 1-3. fail image 불러오기
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/wheeze' # 폴더경로
os.chdir(path) # 폴더로 이동
fail_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기

i=1
for fail_file in fail_files:
    src = os.path.join(path, fail_file)
    dst = 'fail'+ str(i) + '.png'
    dst = os.path.join(path, dst)
    os.rename(src, dst)
    i+=1

fail_files

png_fail = []
for fail_file in fail_files:
    if '.png' in fail_file:
        f = cv2.imread(fail_file)
        png_fail.append(f)

png_fail
png_scale = cv2.imread(png_fail[0])/255
png_scale.shape























## 데이터 전처리
## imageDataGenerator
train_dir = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail/train'
val_dir = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail/val'
test_dir = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail/test'


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # 폴더 지정
        target_size=(150,150), # 사진 크기 지정
        batch_size=20,
        class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
        test_dir, # 폴더 지정
        target_size=(150,150), #사진크기지정
        batch_size=20,
        class_mode='binary'
)





## 합성곱 신경망
# sequential()모델

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(160,120,3)))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# keras functional API
input_shape = (120,160,3)
image_input = layers.Input(shape=input_shape)
output1 = layers.Conv2D(kernel_size=(3,3), filters=32, activation='relu')(image_input)
output2 = layers.MaxPool2D((2,2))(output1)
output3 = layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu')(output2)
output4 = layers.MaxPool2D((2,2))(output3)
output5 = layers.Conv2D(kernel_size=(3,3), filters=128, activation='relu')(output4)
output6 = layers.MaxPool2D((2,2))(output5)
output7 = layers.Conv2D(kernel_size=(3,3), filters=128, activation='relu')(output6)
output8 = layers.MaxPool2D((2,2))(output7)
output9 = layers.Flatten()(output8)
output10 = layers.Dropout(0.5)(output9)
output11 = layers.Dense(512, activation='relu')(output10)
predictions = layers.Dense(2, activation='softmax')(output11)

model = keras.Model(inputs=image_input, outputs=predictions)
model.summary()

# 모델 컴파일 & 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
model.fit(train_generator, test_generator, epochs=10, batch_size=100)


