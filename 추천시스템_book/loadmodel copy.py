
#
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

df_hybrid = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/df_hybrid.csv')
df_hybrid = pd.DataFrame(df_hybrid)
# Split train- & testset
n = 100000
df_hybrid = df_hybrid.sample(frac=1).reset_index(drop=True)
df_hybrid_train = df_hybrid[:100000]
df_hybrid_test = df_hybrid[-n:]

df_hybrid_train[['User']].values.astype(int).tolist()
df_hybrid_train[['Movie']].values.astype(int).tolist()
df_hybrid_test[['User']].values.astype(int).tolist()
df_hybrid_test[['Movie']].values.astype(int).tolist()



train_tfidf_df = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/train_tfidf_df.csv')
# test_tfidf_df = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/test_tfidf_df.csv')


train_tfidf_df[['User','Movie']].tolist().astype(int)
# test_tfidf_df[['User','Movie']].tolist().astype(int)



# 모델 복원
loaded_model = tf.keras.models.load_model('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/hybrid_deeplearning_movie3.h5')
loaded_model.summary()

y_pred = loaded_model.predict([df_hybrid_test['User'], df_hybrid_test['Movie'], train_tfidf_df])
y_true = df_hybrid_test['Rating'].values

mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))
