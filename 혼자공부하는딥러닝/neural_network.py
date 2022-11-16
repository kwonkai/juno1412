# 1. import library
# tensorflow 가 keras를 가져와서 자동으로 keras 적용됨
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np

# 2. open fashion mnist dataset
def open_data():
    (train_input, train_target), (test_input, test_target) = tf.keras.datasets.fashion_mnist.load_data()
    # train data check
    print('train_input :', train_input.shape)
    print('train_target :',train_target.shape)

    # test data check
    print('test_input :',test_input.shape)
    print('test_target :',test_target.shape)    
    
    return (train_input, train_target), (test_input, test_target)


# 3. fashion mnist dataset check
def open_fashion(data):
    fig, axs = plt.subplot(1, 10, figsize=(10,10))
    for i in range(10):
        axs[i].imshow(data[i], cmap='gray_r')
        axs[i].axis('off')
    return plt.show()

# 4. logistic regression classification
# data normailization # 데이터 0~1사이 값으로 변경
def data_normailization():
    train_scaled = train_input / 255.0
    train_scaled = train_scaled.reshape(-1, 28*28)
    print(train_scaled.shape)
    return train_scaled

def neural_network():
    dense = tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
    model = tf.keras.Sequential(dense)
    model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(train_scaled, train_target, epochs=10)
    model.evaluate(val_scaled, val_target)
    return model





## main
# open dataset
(train_input, train_target), (test_input, test_target) = open_data()

# fashion mnist dataset check
image_check = open_fashion(train_input)

# data normailization
train_scaled = data_normailization()

# train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size= 0.2, random_state = 42)

# model build & train & evaluate
model = neural_network()
