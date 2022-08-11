# Hybrid 추천 - CF + MF

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Split train- & testset
# 시드값 고정
import tensorflow as tf
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# Load a movie metadata dataset
movie_metadata = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/movies_metadata.csv', low_memory=False)[['id', 'title', 'genres', 'overview']].drop([19730, 29503, 35587], axis=0).dropna()

# id 컬럼 type 변경
movie_metadata['id'] = movie_metadata['id'].astype(float).astype(int)

## rating 데이터 불러오기 & id기준으로 병합하기
rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/ratings.csv').rename(columns={"movieId" : "id"}).drop('timestamp', axis=1)
hybrid_df = pd.merge(rating, movie_metadata, on='id')
# Split train- & testset
# n = 100000
hybrid_df_genre = hybrid_df.sample(frac=1)[:100000]


## 장르 벡터화
########### 장르 전처리 #######
# 장르 -> [0,1,0,1,0,0,1,.....]
## 중첩리스트 제거 & 방송국 지우기
from ast import literal_eval 
hybrid_df_genre= hybrid_df_genre.copy()
hybrid_df_genre['genres'] =hybrid_df_genre['genres'].apply(literal_eval)
hybrid_df_genre['genre_name'] = hybrid_df_genre['genres'].apply(lambda x : [ y['name'] for y in x])


# df_id_overview = hybrid_df_50[['id', 'overview']].set_index('id')


genre_feature = pd.DataFrame(hybrid_df_genre['genre_name'].to_list())
# genre_feature.replace({'Animation':1},{'Adventure':2},{'Romance':3},{'Comedy':4},{'Family':5},{'History':6})
# genre_feature.replace({'Crime':8},{'Fantasy':9},{'Science Fiction':10},{'Thriller':11},{'Music':12},{'Horror':13},{'Documentary':14})
# genre_feature.replace({'Mystery':15},{'Western':16},{'TV Movie':17},{'War':18},{'Foreign':19},{'Drama':7})

genre_feature.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7']



remove_list = genre_feature.isin(['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions',\
                                   'Aniplex', 'GoHands', 'BROSTA TV', 'Mardock Scramble Production Committee', 'Sentai Filmworks', \
                                   'Odyssey Media', 'Pulser Productions', 'Rogue State', 'The Cartel'])
               

# genre 유일값 확인하기 : 방송국 제거되었는지 확인
print(pd.unique(genre_feature[['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7']].values.ravel()))


genre_feature['genre_feature'] = genre_feature.fillna('').apply(lambda x:','.join(x), axis=1)


hybrid_df = hybrid_df_genre.reset_index().join(genre_feature['genre_feature']).set_index('index').drop(columns=['overview','genre_name', 'genres'])
hybrid_df.columns = ['userId', 'movieId', 'rating', 'title', 'genre_feature']

# mapping 중요!! 
# 무조건 있어야 할 것!
# Create user- & movie-id mapping
user_id_mapping = {id:i for i, id in enumerate(hybrid_df['userId'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(hybrid_df['movieId'].unique())}

# Use mapping to get better ids
hybrid_df['userId'] = hybrid_df['userId'].map(user_id_mapping)
hybrid_df['movieId'] = hybrid_df['movieId'].map(movie_id_mapping)

# train test 분리
ratings = shuffle(hybrid_df, random_state=1)
# cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.reset_index().drop(columns=['index'],axis=0)[:80000]
ratings_test = ratings.reset_index().drop(columns=['index'],axis=0)[-20000:]

# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

##### CF 추천 알고리즘 >>>>>>>>>>>>>>>

rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# train set 사용자들의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# train 데이터의 user의 rating 평균과 영화의 평점편차 계산 
rating_mean = rating_matrix.mean(axis=1)
rating_bias = (rating_matrix.T - rating_mean).T

def CF_knn_bias(userId, movieId, neighbor_size=0):
    if movieId in rating_bias:
        sim_scores = user_similarity[userId]
        movie_ratings = rating_bias[movieId]
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        if neighbor_size == 0:
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            prediction = prediction + rating_mean[userId]
        else:
            if len(sim_scores) > 1:
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                user_idx = np.argsort(sim_scores)
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                prediction = prediction + rating_mean[userId]
            else:
                prediction = rating_mean[userId]
    else:
        prediction = rating_mean[userId]
    return prediction


##### MF 추천 알고리즘 >>>>>>>>>>>>>>>

class NEW_MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose=True):
        self.R = np.array(ratings)
        item_id_index = []
        index_item_id = []
        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose

    # train set의 RMSE 계산
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    # Ratings for user i and item j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Test set을 선정
    def set_test(self, ratings_test):
        test_set = []
        for i in range(len(ratings_test)):
            x = self.user_id_index[ratings_test.iloc[i, 0]]
            y = self.item_id_index[ratings_test.iloc[i, 1]]
            z = ratings_test.iloc[i, 2]
            test_set.append([x, y, z])
            self.R[x, y] = 0                    # Setting test set ratings to 0
        self.test_set = test_set
        return test_set                         # Return test set

    # Test set의 RMSE 계산
    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))

    # Training 하면서 test set의 정확도를 계산
    def test(self):
        # Initializing user-feature and item-feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        # List of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i,j]) for i, j in zip(rows, columns)]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i+1, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %.4f" % (i+1, rmse1, rmse2))
        return training_process

    # Ratings for given user_id and item_id
    def get_one_prediction(self, user_id, item_id):
        prediction = self.get_prediction(self.user_id_index[user_id], self.item_id_index[item_id])
        return prediction

    # Full user-movie rating matrix
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)

# MF클래스 생성 및 학습
R_temp = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
mf = NEW_MF(R_temp, K=200, alpha=0.001, beta=0.02, iterations=100, verbose=True)
test_set = mf.set_test(ratings_test)
result = mf.test()


##### Hybrid 추천 알고리즘

def recommender0(recomm_list, mf):
    recommendations = np.array([mf.get_one_prediction(user, movie) for (user, movie) in recomm_list])
    return recommendations

def recommender1(recomm_list, neighbor_size=0):
    recommendations = np.array([CF_knn_bias(user, movie, neighbor_size) for (user, movie) in recomm_list])
    return recommendations

recomm_list = np.array(ratings_test.iloc[:, [0, 1]])
predictions0 = recommender0(recomm_list, mf)
RMSE2(ratings_test.iloc[:, 2], predictions0)
predictions1 = recommender1(recomm_list, 37)
RMSE2(ratings_test.iloc[:, 2], predictions1)

weight = [0.8, 0.2]
predictions = predictions0 * weight[0] + predictions1 * weight[1]
RMSE2(ratings_test.iloc[:, 2], predictions)


y_pred = predictions
y_true = ratings_test['rating'].values
test = y_pred.reshape(25000,)
tmp = np.stack((y_true, test), axis=1)
print(tmp)
df_v = pd.DataFrame(tmp, columns=['true', 'pred'])
df_v.head(10)

result = df_v.sort_values(by=['true', 'pred'])
result =result.reset_index(drop=True)

# 그래프 그리기
import matplotlib.pyplot as plt

for i in range(len(y_pred)):
    if result['true'][i] > result['pred'][i]:
        plt.vlines(i, result['pred'][i], result['true'][i], color='gray', linestyle='solid', linewidth=1)
    else:
        plt.vlines(i, result['true'][i], result['pred'][i], color='gray', linestyle='solid', linewidth=1)
plt.xlabel('rating_count')
plt.ylabel('rating')

plt.savefig('new_metadata_bookmodel_deeplearning_test2.png')

# 그래프 크기 계산
y_pred = result['pred']
y_area = []
y_area1 = []
y_area2 = []
for i in range(len(y_pred)):
    if result['true'][i] > result['pred'][i]:
        y_area1_list = result['true'][i] - result['pred'][i]
        y_area1.append(y_area1_list)
    else:
        y_area2_list = result['pred'][i] - result['true'][i]
        y_area2.append(y_area2_list)
    y_area = y_area1 + y_area2

print(sum(y_area))

for i in np.arange(0, 1, 0.01):
    weight = [i, 1.0 - i]
    predictions = predictions0 * weight[0] + predictions1 * weight[1]
    print("Weights - %.2f : %.2f ; RMSE = %.7f" % (weight[0], 
           weight[1], RMSE2(ratings_test.iloc[:, 2], predictions)))

#######################################################################
# y_pred = model.predict([df_hybrid_test['userId'], df_hybrid_test['movieId'], test_tfidf_title.toarray()])
# y_true = df_hybrid_test['rating'].values
# y_pred = np.where(y_pred < 0.5, 0.5, np.where(y_pred > 5, 5, y_pred))


# test = y_pred.reshape(20000,)
# tmp = np.stack((y_true, test), axis=1)
# print(tmp)
# df_v = pd.DataFrame(tmp, columns=['true', 'pred'])
# df_v.head(10)

# result = df_v.sort_values(by=['true', 'pred'])
# result =result.reset_index(drop=True)
# result.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/origin_overview_genre_test.csv')


# 그래프 그리기
# y_pred = result['pred']
# for i in range(len(y_pred)):
#     if result['true'][i] > result['pred'][i]:
#         plt.vlines(i, result['pred'][i], result['true'][i], color='red', linestyle='solid', linewidth=1)
#     else:
#         plt.vlines(i, result['true'][i], result['pred'][i], color='red', linestyle='solid', linewidth=1)
# plt.xlabel('rating_count')
# plt.ylabel('rating')

# plt.savefig('movielens_100k_originmodel_test.png')

# # 그래프 그리기
# y_pred = result['pred']
# y_area = []
# y_area1 = []
# y_area2 = []
# for i in range(len(y_pred)):
#     if result['true'][i] > result['pred'][i]:
#         y_area1_list = result['true'][i] - result['pred'][i]
#         y_area1.append(y_area1_list)
#     else:
#         y_area2_list = result['pred'][i] - result['true'][i]
#         y_area2.append(y_area2_list)
#     y_area = y_area1 + y_area2

# print(sum(y_area))

# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_log_error
# from sklearn.metrics import r2_score

# mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
# rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
# # rmsle = np.sqrt(mean_squared_log_error(y_pred=y_pred, y_true=y_true))/
# mae = mean_absolute_error(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)

# print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
# print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))
# # print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} RMSLE'.format(rmsle))
# print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} MAE'.format(mae))
# print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} R2'.format(r2))


# rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
# print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))