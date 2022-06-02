# 히트맵 시각화

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Dense, Dropout
from keras_preprocessing import image
from tensorflow.keras import backend as K
from distutils.errors import PreprocessError


# cnn modeling
def cnn():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(240,320,3)))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(layers.Conv2D(128,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(layers.Flatten())
    model.add(Dropout(0.3))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = cnn()
model.summary()


## 이미지 입력 및 전처리



img_path = '/home/juno/workspace/codetrain/trainset/test/fail/fail656.png'
# target PIL(Python Image Library)
img_fail = image.load_img(img_path, target_size=(240,320)) 

# (240, 320, 3)에 numpy float32 배열
img_x = image.img_to_array(img_fail)

# 차원 추가 3d -> 4d
img_x_4d = np.expand_dims(img_x, axis=0)

# 정규화
img_x_4d /= 255.


plt.imshow(img_x_4d[0])


## Grad_CAM 알고리즘
# 층 가져오기 = model cnn 마지막층
last_conv_layer = model.get_layer('conv2d_5')

grads = K.gradients(model.output, last_conv_layer)[0]

pooled_grads = K.mean(grads, axis=(0,1,2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0], last_conv_layer.output])

pooled_grads_value, conv_layer_output_value, output = iterate([img_x_4d])

print(pooled_grads_value.shape, conv_layer_output_value.shape, output.shape)


for i in range(128):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
# 만들어진 특성 맵에서 채널 축을 따라 평균한 값이 클래스 활성화의 히트맵이다.

heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)

plt.matshow(heatmap)
plt.show()

