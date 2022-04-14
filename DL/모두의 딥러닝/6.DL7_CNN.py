# CNN

# 라이브러리 설정
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기 & 데이터셋
(x_train, y_class_train), (x_test, y_class_test) = mnist.load_data()

print("학습셋 이미지 수 : %d 개" % (x_train.shape[0]))
print("테스트세 이미지 수 : %d 개" % (x_test.shape[0]))

plt.imshow(x_train[0], cmap='Greys')
plt.show()

# 코드로 확인
for x in x_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

# 차원 변화 과정
x_train = x_train.reshape(x_train.shape[0], 784)
x_train = x_train.astype('float64')
x_train = x_train/255

x_test = x_test.reshape(x_test.shape[0], 784).astype('float64') / 255

# 클래스 값 확인
print("class : %d" % (y_class_train[0]))

# 바이너리화 과정
y_train = np_utils.to_categorical(y_class_train, 10)
y_test = np_utils.to_categorical(y_class_test, 10)

print(y_train[0])
