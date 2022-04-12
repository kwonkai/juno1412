# keras 함수 불러오기
from keras.models import Sequential
from keras.layers import Dense

# 라이브러리 불러오기
import numpy
import tensorflow as tf
tf.__version__

# seed=0 설정 -> 실행할 때마다 같은 결과 출력
seed=0
numpy.random.seed(seed)
tf.set_random_seed(seed)

