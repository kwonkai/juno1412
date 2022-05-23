# 코드 트레이닝 폐소리 데이터 구분하기


from cgitb import text
from string import digits
import string
import pandas as pd
import numpy as np
import os
from PIL import Image 
from glob import glob
import cv2
import tensorflow as keras
import matplotlib.pyplot as plt
from string import digits

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
none_files[5]

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




