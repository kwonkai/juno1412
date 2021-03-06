# 코드 트레이닝 폐소리 데이터 구분하기


from cgitb import text
from string import digits
import string
import pandas as pd
import numpy as np
import os
import PIL
from PIL import Image 
from glob import glob
import cv2
from sklearn.preprocessing import LabelEncoder
import tensorflow as keras
import matplotlib.pyplot as plt
from string import digits

## 3.데이터 train/test/valudation set 분류
import os, shutil

# 1. 모든데이터
# 1-1. 성공/실패 데이터 폴더 만들기
# global 변수
global data_path, none_path, fail_path

data_path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail' # 폴더경로
none_path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none'
fail_path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/wheeze'

# 1-1. 파일명 변경하기
def change_name(name):
    if name == 'none':
        os.chdir(none_path) # 폴더로 이동
        none_files = os.listdir(none_path) # 해당 폴더에 있는 파일 이름 리스트 받기
        i=1
        for none_file in none_files:
            src = os.path.join(none_path, none_file)
            dst = 'none'+ str(i) + '.png'
            dst = os.path.join(none_path, dst)
            os.rename(src, dst)
            i+=1
            
    
    elif name == 'fail':
        os.chdir(fail_path) # 폴더로 이동
        fail_files = os.listdir(fail_path) # 해당 폴더에 있는 파일 이름 리스트 받기
        i=1
        for fail_file in fail_files:
            src = os.path.join(fail_path, fail_file)
            dst = 'fail'+ str(i) + '.png'
            dst = os.path.join(fail_path, dst)
            os.rename(src, dst)
            i+=1
    return 


# 1-2훈련/검증/테스트 분할 폴더 생성 + 파일 복사
# 함수화, 줄여야함...
def up_separate(dataset):
    if dataset == 'train':
        train_pic = os.path.join(data_path, 'train')
        os.mkdir(train_pic)
        train_none_dir = os.path.join(train_pic, 'none')
        os.mkdir(train_none_dir)
        train_fail_dir = os.path.join(train_pic, 'fail')
        os.mkdir(train_fail_dir)

    elif dataset == 'val':
        val_pic = os.path.join(data_path, 'val')
        os.mkdir(val_pic)
        val_none_dir = os.path.join(val_pic, 'none')
        os.mkdir(val_none_dir)
        val_fail_dir = os.path.join(val_pic, 'fail')
        os.mkdir(val_fail_dir)

  
    elif dataset == 'test':
        test_pic = os.path.join(data_path, 'test')
        os.mkdir(test_pic)
        test_none_dir = os.path.join(test_pic, 'none')
        os.mkdir(test_none_dir) 
        test_fail_dir = os.path.join(test_pic, 'fail')
        os.mkdir(test_fail_dir) 
    

    def copy_files():
        src = os.path.join(data_path, pic)
        for pic in pics:
            pics = ['none{}.png'.format(i) for i in range(1,1500)]
            dst = os.path.join(train_none_dir, pic)
            shutil.copy(src,dst)

        pics = ['none{}.png'.format(i) for i in range(1500, 2200)]
        for pic in pics:
            dst = os.path.join(val_none_dir, pic)
            shutil.copy(src,dst)

        pics = ['none{}.png'.format(i) for i in range(2200, 2800)]
        for pic in pics:
            dst = os.path.join(test_none_dir, pic)
            shutil.copy(src,dst)


        # fail 데이터
        pics = ['none{}.png'.format(i) for i in range(1,300)]
        for pic in pics:
            dst = os.path.join(train_fail_dir, pic)
            shutil.copy(src,dst)

        pics = ['none{}.png'.format(i) for i in range(300, 400)]
        for pic in pics:
            dst = os.path.join(val_fail_dir, pic)
            shutil.copy(src,dst)

        pics = ['none{}.png'.format(i) for i in range(400, 515)]
        for pic in pics:
            dst = os.path.join(test_fail_dir, pic)
            shutil.copy(src,dst)

        return copy_files

        
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(none_files[0])

# 2-2. label 변환
label = []
for n, path in enumerate(none_files[:100]):
    token = text_to_word_sequence[none_files[n]]
    label.append[token(0)]



# 파일 확인 - 완료
# OPENCV
img = cv2.imread('C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none/none14.png')
print(img.shape)
img_resize = cv2.resize(img, (20,20))
img_resize_1 = img_resize.flatten() # 1차원 변경 - 완료
img_resize_1.shape


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



# # 데이터 1차원 변경하기 - 폴더

# # 2-1. 텍스트 토큰화
# none_files = glob('*.png')
# none_files

# from keras.preprocessing.text import text_to_word_sequence
# text_to_word_sequence(none_files[0])

# # 2-2. label 변환
# label = []
# for n, path in enumerate(none_files[:100]):
#     token = text_to_word_sequence(none_files[n])
#     label.append(token[0])

# items = label
# encoder = LabelEncoder()
# encoder.fit(items)
# label = encoder.transform(items)
# encoder.inverse_transform(label)

# ##  이미지 변환
# image = cv2.imread(none_files[0])/255
# image.shape

# path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none' # 폴더경로
# os.chdir(path) # 폴더로 이동
# none_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기
# format = ['.png']
# none_files[5]

# # i = 1
# # for none_file in none_files:
# #     img = Image.open(path, none_file[i])
# #     img_resize = cv2.resize(img, (20,20))
# #     img_flat = img_resize.flatten()
# #     i += 1
# #     print(img_flat)

# none_files = none_files.glob('none/*')
# PIL.Image.open(str(none_files[0]))


# 1. 데이터 불러오기
# 1-1. both 데이터 불러오기
# 1-2. none image 불러오기
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none' # 폴더경로
os.chdir(path) # 폴더로 이동
none_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기

# 파일 이름 변경하기
i=1
for none_file in none_files:
    src = os.path.join(path, none_file)
    dst = 'none'+ str(i) + '.png'
    dst = os.path.join(path, dst)
    os.rename(src, dst)
    i+=1

none_files = none_files
none_files[:10]

a = none_files[0]


## 데이터셋 만들기
batch_size = 50
height = 160
width = 120


#  png 파일 불러오기
# none_datas = glob('*.png')
# none_datas =','.join((x for x in none_datas if not x.isdigit()))
# len(none_datas)
# none_datas

# 2-1. 텍스트 토큰화
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(none_files[0])

# 2-2. label 변환
label = []
for n, path in enumerate(none_files[:100]):
    token = text_to_word_sequence[none_files[n]]
    label.append[token(0)]



# 1-3. fail image 불러오기
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/wheeze' # 폴더경로
os.chdir(path) # 폴더로 이동
fail_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기

# #  png 파일 불러오기
# fail_datas = glob('*.png')
# fail_datas =','.join((x for x in fail_datas if not x.isdigit()))
# len(fail_datas)
# fail_datas

i=1
for fail_file in fail_files:
    src = os.path.join(path, fail_file)
    dst = 'fail'+ str(i) + '.png'
    dst = dst.resize(img, (80, 60))
    dst = os.path.join(path, dst)
    os.rename(src, dst)
    i+=1

fail_files



# 2. 텍스트 원핫인코딩
# 2-1. 텍스트 토큰화
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(none_datas)







# table = str.maketrans('', '', digits)
# newstring = string.translate(table)


# none_datas = [x for x in none_datas if 'png' in x]
# none_datas
# len(none_datas)

none = []
for none_data in none_datas:
    data = Image.open(none_data)
    data = data.resize((320, 240))
    data = np.array(data)
    none.append(data)

x_train = none.reshape(none.shape[0], 320, 240, 1).astype('float64') / 255





# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float64') / 255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float64') / 255
# Y_train = to_categorical(Y_train, 10)
# Y_test = to_categorical(Y_test, 10)

# path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none' # 폴더경로
# os.chdir(path) # 폴더로 이동
# none_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기
# none_files

# png_img = []
# for file in none_files:
#     if '.png' in file: 
#         f = cv2.imread(file)
#         png_img.append(f)
        
# png_img[:5]
# all_data = glob('C:/Users/kwonk/juno1412-1/juno1412/DL/code_training')




