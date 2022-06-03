import tensorflow as tf
from tensorflow.keras.preprocessing import image

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pickle

from PIL import Image


### 예측 함수
def model_predict(model, class_index, label):
    files_name=[]
    predics=[]
    labels=[]

    for i in range(len(class_index)):
        if i % 100 == 0:
          print(i, end=', ')
        test_path = '/content/gdrive/My Drive/dataset/test/{}/{}'.format(class_index[i][:3], class_index[i])
        test_image = Image.open(test_path).resize((224, 224))

        img_tensor = np.array(test_image)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        test_data = img_tensor / 255.0

        pred = model.predict(test_data)

        files_name.append(class_index[i])
        predics.append(pred)
        labels.append(label)

    re_dic = dict({'files':files_name, 'pred':predics, 'labels':labels})

    return re_dic


### Grad_CAM
def grad_cam(model, class_predict):
    plt.figure(figsize=(32, 32))

    for i in range(len(class_predict['pred'])): 
        path = os.path.join('/content/gdrive/My Drive/dataset/test/{}/{}'.format(class_predict['files'][0][:3], class_predict['files'][i]))
        image = cv2.imread(path)
        img = cv2.resize(image, (224, 224))
        x = img.copy()
        x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
          
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(index=23).output, model.output]
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
        weights = weights.reshape(512, 1)

        cam = (prediction -0.5) * np.matmul(model_outputs, weights)
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam -= 0.2
        cam /= 0.8

        try:
          cam = cv2.resize(np.float32(cam), (224, 224))
        except Exception as e: 
          print(str(e))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.2)] = 0
        grad_cam = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
        plt.axis('off')
        plt.imshow(grad_cam[:, :, ::-1])
        plt.savefig('/content/gdrive/My Drive/dataset/Grad_CAM/{}/dog_0/{}'.format(class_predict['files'][i][:3], class_predict['files'][i]))
        plt.close()


## 모델 예측 데이터 가져오기
test_path = '/content/gdrive/My Drive/dataset/test/'

cat_path = os.path.join(test_path, os.listdir(test_path)[0])
dog_path = os.path.join(test_path, os.listdir(test_path)[1])

cat_files = os.listdir(cat_path)
dog_files = os.listdir(dog_path)


## 학습 모델 가져오기
get_model = tf.keras.models.load_model('/.h5')

## 예측 분류하기
# 1. 고양이
cats_predict = model_predict(get_model, cat_files, 0)
for i in range(10):
    print(cats_predict['files'][i], cats_predict['pred'][i][0][0], cats_predict['labels'][i])

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

print('맞은 것')
for i in range(10):
    print(cat_0['files'][i], cat_0['pred'][i])
print('틀린 것')
for i in range(10):  
    print(cat_1['files'][i], cat_1['pred'][i])

