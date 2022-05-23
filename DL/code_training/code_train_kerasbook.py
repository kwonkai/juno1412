# 코드 트레이닝 폐소리 데이터 구분하기


from cgitb import text
import pickletools
import pandas as pd
import numpy as np
import os
from PIL import Image 
from glob import glob
import cv2
import tensorflow as keras
import matplotlib.pyplot as plt
from string import digits
import shutil

# 파일 확인 - 완료
# OPENCV
img = cv2.imread('C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none/none14.png')
print(img.shape)
img_resize = cv2.resize(img, (20,20))
img_resize_1 = img_resize.flatten() # 1차원 변경 - 완료
img_resize_1.shape

plt.imshow(img_resize_1, cmap='gray'), plt.axis("off")
plt.show()

# 데이터 1차원 변경하기 - 폴더
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none' # 폴더경로
os.chdir(path) # 폴더로 이동
none_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기
format = ['.png']
len(none_files)

i = 1
for none_file in none_files:
    img = Image.open(path, none_file[i])
    img_resize = cv2.resize(img, (20,20))
    img_flat = img_resize.flatten()
    i += 1
    print(img_flat)


# 1. 데이터 불러오기
# 1-1. both 데이터 불러오기
# 1-2. none image 불러오기
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none' # 폴더경로
os.mkdir("C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail")
os.chdir(path) # 폴더로 이동
none_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기

# 파일 이름 변경하기
i=1
for none_file in none_files:
    src = os.path.join(path, none_file)
    dst = 'none'+ str(i) + '.png'
    dst = os.path.join(path, dst)
    os.rename(src, dst)
    shutil.copy(none_files, './C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail')
    i+=1

none_files = none_files


# 1-3. fail image 불러오기
path = 'C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/wheeze' # 폴더경로
os.mkdir("C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail")
os.chdir(path) # 폴더로 이동
fail_files = os.listdir(path) # 해당 폴더에 있는 파일 이름 리스트 받기

i=1
for fail_file in fail_files:
    src = os.path.join(path, fail_file)
    dst = 'fail'+ str(i) + '.png'
    dst = os.path.join(path, dst)
    os.rename(src, dst)
    shutil.copy(fail_file[i], './C:/Users/kwonk/juno1412-1/juno1412/DL/code_training/datasets_lungsound/none_fail')
    i+=1

fail_files = fail_files




# 2-1. 텍스트 토큰화
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(none_files[0])

# 2-2. label 변환
label = []
for n, path in enumerate(none_files[:100]):
    token = text_to_word_sequence[none_files[n]]
    label.append[token(0)]




import os, shutil

# 1. train data
data_path = 'C:\\Users\\kwonk\\juno1412-1\\juno1412\\DL\\code_training\\datasets_lungsound\\none_fail' # 폴더경로
os.chdir(data_path) # 폴더로 이동
none_files = os.listdir(data_path) # 해당 폴더에 있는 파일 이름 리스트 받기

# 훈련/검증/테스트 분할 폴더
# 훈련
train_pic = os.path.join(data_path, 'train')
os.mkdir(train_pic)
val_pic = os.path.join(data_path, 'val')
os.mkdir(val_pic)
test_pic = os.path.join(data_path, 'test')
os.mkdir(test_pic)

# none/fail의 train/validation/test 디렉터리 분리
# train
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
len(fail_files)
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
