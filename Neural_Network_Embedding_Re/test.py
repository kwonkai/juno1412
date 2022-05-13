from IPython.core.interactiveshell import InteractiveShell

# 파일 다운로드
# utils import get_file == url -> file download
import tensorflow as tf
# from keras.utils import get_file

x = tf.keras.utils.get_file('found_books_filtered.ndjson', 'https://raw.githubusercontent.com/WillKoehrsen/wikipedia-data-science/master/data/found_books_filtered.ndjson')

# books = list만들기
import json

books = []

with open(x, 'r') as fin: # 'r' 읽기용으로 파일 열기
    # Append each line to the books
    books = [json.loads(l) for l in fin]

# Remove non-book articles
books_with_wikipedia = [book for book in books if 'Wikipedia:' in book[0]]
books = [book for book in books if 'Wikipedia:' not in book[0]]
print(f'Found {len(books)} books.')


## 데이터 전처리
# 책정보 정수로 변경하기 # index 
book_index = {book[0] : idx for idx, book in enumerate(books)} #enumerate : 인덱스, 원소로 이루어진 tuple로 만들어줌
index_book = {idx : book for book, idx in book_index.items()} # items() : key와 대응값 가져오기 # book_index의 대응값 'title' 가져오기

# Exploring Wikilinks
# chain method = 자기자신을 반환하면서 다른 함수를 지속적으로 호출할 수 있는 방법
from itertools import chain
wikilinks = list(chain(*[book[2] for book in books]))


# 가장 많이 연결된 기사 찾기
# items 항목 수가 카운트된 dictionary를 반환하는 함수를 만든다.
# collections module : count(개수세기), OrderedDict

from collections import Counter, OrderedDict

def count_items(l):
  # Return ordered dictionary of counts of objects in `l`
  # create a count object
  counts = Counter(l)

  # sort by highest count first and place in orderd dictionary
  # sort(key = (key인자에 함수를 넘겨주면 우선순위가 정해진다))
  counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)  # x[1] 우선순위 숫자로 변경, reverse = 높은 숫자부터
  counts = OrderedDict(counts) # 데이터 순서 설정(key, val)

  return counts


# Find set of wikilinks from each book and convert to a flattend last
# 각각 책에서 wikilinks 설정을 찾고 1차원으로 변경하기
unique_wikilinks = list(chain(*[list(set(book[2])) for book in books])) # books의 중복치를 제거한 wikilinks 값
wikilinks = [link.lower() for link in unique_wikilinks] # lower() 대문자 -> 소문자 : 동일링크 : paperback, Paperback, PAPERBACK 등 링크 통합

wikilink_counts = count_items(wikilinks) # 가장 많이 사용된 wikilinks의 unique_counts 상위 10개 불러오기
list(wikilink_counts.items())[:10]


to_remove = ['hardcover', 'paperback', 'hardback', 'e-book', 'wikipedia:wikiproject books', 'wikipedia:wikiproject novels'] 

for t in to_remove:
    wikilinks.remove(t)
    _ = wikilink_counts.pop(t) # ????? #pop(t) t가 들어간 to_move의 카테고리들을 제거해라


# 5번 이상 나온 wikilinks를 사용한다.
links = [t[0] for t in wikilink_counts.items() if t[1] >= 5] # ?????
type(links)
len(links)

# links indexing
link_index = {link: idx for idx, link in enumerate(links)}
index_link = {idx: link for link, idx in link_index.items()}


# 데이터 전처리 결과
print(f'Found {len(books)} books.')
print(f'Found {len(links)} links.')


## train set making
pairs = [] # pairs 빈 list 생성

# 각 책에대한 반복 수행
for book in books:
    # 각 책에대한 링크를 반복 수행
    # 720,000개의 예시 추가
    # 예시 각 title마다 link가 들어간 pairs만들기 (2, 616), (2, 2914) -> 77만개
    # link_index[link.lower()]) for link in book[2] if link.lower() in links) -> links의 link가 소문자배열이라면 book[2]의 link를 반복해서 꺼내준다. -> link_index에 link.lower()= ex)navi 값이 들어가면 index 값이 나오게 된다.
    pairs.extend((book_index[book[0]], link_index[link.lower()]) for link in book[2] if link.lower() in links)

# paris 중복 제거
pairs_set = set(pairs)

# 가장 자주 나타나는 (title, link)
x = Counter(pairs)
sorted(x.items(), key = lambda x: x[1], reverse = True)[:10]

# 데이터셋 생성기
# 데이터셋 positive, negative 생성기 만들기
# batch_size 정하기

import numpy as np
import random
random.seed(100)

def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
  # batch를 저장할 numpy 배열 준비하기
  batch_size = n_positive * (1 + negative_ratio) # 1 : positive가 할당받는 1 # + n_positive * (1 + negative_ratio) = n_positive + negative_ratio 
  batch = np.zeros((batch_size, 3)) # shape = batch_size * 3 -> next batch (book, link, batch)

  # 라벨 조정하기
  if classification:
    neg_label = 0
  else:
    neg_label = -1

  # 생성기 만들기
  while True:
    # 랜덤 positive 예시 선택
    for idx, (book_id, link_id) in enumerate(random.sample(pairs, n_positive)):
      batch[idx, :] = (book_id, link_id, 1)

    # idx = 1씩 증가
    idx += 1

    # batchsize가 찰때까지, negative examples 추가
    while idx < batch_size:

      # 랜덤선택
      random_book = random.randrange(len(books))
      random_link = random.randrange(len(links))

      # positive sample이 아니라는 걸 체크
      if (random_book, random_link) not in pairs_set:

        # 배치에 negative_index  추가하기 
        batch[idx, :] = (random_book, random_link, neg_label)
        idx += 1

      
    # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'book': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]



## Nerual Network Embedding Model
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Dense
from tensorflow.keras.models import Model

def book_embedding_model(embedding_size=50, classification = False):

  # """Model to embed books and wikilinks using the functional API.
  #    \Trained to discern if a link is present in a article"""

    # 1차원 입력
    book = Input(name='book', shape=[1])
    link = Input(name='link', shape=[1])

    # 책 Embedding(None, 1, 50)
    # book Embeding (None, input, output)(book)
    # 30720 x 50 = 2,087,900 param(노드에 연결된 간선 수)
    book_embedding = Embedding(name = 'book_embedding',
                               input_dim = len(book_index), #book_index = 1
                               output_dim = embedding_size)(book) # book을 embedding해야함으로 뒤에 (book) +

    # link Embedding(None, 1, 50)
    # link Embeding (None, input, output)(link)
    # 41758 x 50 = 2,087,900 param(노드에 연결된 간선 수)
    link_embedding = Embedding(name = 'link_embedding',
                               input_dim = len(link_index),
                               output_dim = embedding_size)(link)

    

    # 내적으로 book&link embedding 1개의 Embedding으로 변형
    # merged = shape(None, 1, 1)
    # Dot(name, normalize(정규화), axes(샘플 간 내적계산))
    # Dot(normalize = True -> L2 Norm -> 내적 출력은 샘플간의 cosine 근접도 )
    merged = Dot(name = 'dot_product', normalize = True, axes=2)([book_embedding, link_embedding])

    # Reshape to be single Number(shape will be(None, 1))
    # Reshape_shape = 정수튜플/샘플/배치크기 포함하지 않음
    # Regression에 대한 모델 출력
    merged = Reshape(target_shape = [1])(merged)

    # if classifcation, add extra layers and loss function is binary crossentroy
    # Dense layer에서 입력뉴런 수에 상관없이 출력 뉴런 수를 자유롭게 설정, 이진 분류문제에서 0,1을 나타내는 출력뉴런이 하나만 있으면 되기에 출력뉴런 1개, 입력뉴런/가중치 계싼값 0,1로 표현가능한 sigmoid함수를 사용함.
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged) #  분류가 0 또는 1이기 때문에 sigmoid 함수를 사용한다. # 
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

      # Otherwise loss function is mean squared error
    else:
      # model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer='adam', loss='mse')

    return model

# Instantitate model and show parameters
model = book_embedding_model()
model.summary()


## train Embedding model
n_positive = 1024

# negative_ratio = 2 인 이유는 2가 가장 잘 됨 = 제작자
gen = generate_batch(pairs, n_positive, negative_ratio=2)

# Train
# steps_per_epoch = 1epoch마다 사용할 batch_size를 정의함
# steps_per_epoch : 각 epoch에서 동일한 수의 pairs를 볼 수 있게 한다.
# verbose(상세정보) 보통 0, 자세히 1, 함축정보 2
model.fit_generator(gen, epochs = 15, steps_per_epoch = len(pairs) // n_positive, verbose=2)


model.save('first_attempt.h5')


# extract embedding & analyze
# book에 대한 embedding울 추출해 유사한 book&link를 찾는데 사용한다.
# 각 book 50차원 vector로 표시
book_layer = model.get_layer('book_embedding')
book_weights = book_layer.get_weights()[0] # 1차원이라 [0]밖에 없음
book_weights.shape

# embedding이 내적 정규화하여(-1, 1) cosine similarity가 되도록 함
book_weights = book_weights / np.linalg.norm(book_weights, axis = 1).reshape((-1, 1)) # norm = 거리측정, 벡터 사이의 길이 구하기
book_weights[0][:10]
np.sum(np.square(book_weights[0])) #book_weight[0] 제곱 -> sum


## find similar book 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15

def find_similar(name, weights, index_name = 'book', n = 10, least = False, return_dist = False, plot = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    
    # Select index and reverse index
    # rindex() method : 지정 문자열 끝 위치
    if index_name == 'book':
        index = book_index
        rindex = index_book
    elif index_name == 'page':
        index = link_index
        rindex = index_link
    
    # Check to make sure `name` is in index
    try:
        # 대상 책과 모든 책들 사이의 내적계산
        # 임베딩이 정규화된 경우 벡터 간의 내적은 가장 유사하지 않은 -1 에서 가장 유사한 +1 까지의 cosine similarity을 나타냄
        # embedding : 고차원의 벡터공간에서 저차원의 벡터공간으로의 맵핑
        # 임베딩 정규화 : 
        dists = np.dot(weights, weights[index[name]])
    except KeyError: # 책이 없는 경우 not Found
        print(f'{name} Not Found.')
        return
    
    # Sort distance indexes from smallest to largest
    # argsort(오름차순 정렬)
    sorted_dists = np.argsort(dists)
    
    
    # Plot results if specified
    if plot:
        
        # Find furthest and closest items
        furthest = sorted_dists[:(n // 2)]
        closest = sorted_dists[-n-1: len(dists) - 1]
        items = [rindex[c] for c in furthest]
        items.extend(rindex[c] for c in closest)
        
        # Find furthest and closets distances
        distances = [dists[c] for c in furthest]
        distances.extend(dists[c] for c in closest)
        
        colors = ['r' for _ in range(n //2)]
        colors.extend('g' for _ in range(n))
        
        data = pd.DataFrame({'distance': distances}, index = items)
        
        # Horizontal bar chart
        data['distance'].plot.barh(color = colors, figsize = (10, 8),
                                   edgecolor = 'k', linewidth = 2)
        plt.xlabel('Cosine Similarity');
        plt.axvline(x = 0, color = 'k');
        
        # Formatting for italicized title
        name_str = f'{index_name.capitalize()}s Most and Least Similar to'
        for word in name.split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        plt.title(name_str, x = 0.2, size = 28, y = 1.05)
        
        return None


    # 가장 유사하지 않은 book
    if least:
        # 정렬에서 처음 n개 가져오기(유사하지 않는 book)
        closest = sorted_dists[:n]
         
        print(f'{index_name.capitalize()}s furthest from {name}.\n')
        
    # 가장 유사한 book
    else:
        # 정렬에서 맨뒤 n개 가져오기(가장 유사한 book)
        closest = sorted_dists[-n:]
        
        # Need distances later on
        if return_dist:
            return dists, closest
        
        
        print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    # Need distances later on
    if return_dist:
        return dists, closest
    
    
    # 인쇄 길이
    max_width = max([len(rindex[c]) for c in closest])
    
    # 가장 유사한 책, 유사도
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')

# find similar book
# parameter(name='War and Peace, weights='book_weights')
find_similar('War and Peace', book_weights) #
# find similar book 시각화
find_similar('War and Peace', book_weights, n = 10, plot = True)


# wikilinks extract embedding
def extract_weights(name, model):
    """Extract weights from a neural network model"""
    
    # layer, 가중치 받아오기
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    
    # 정규화
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights

link_weights = extract_weights('link_embedding', model)

find_similar('biography', link_weights, index_name = 'page', n = 5, plot = True)


## classification model
model_class = book_embedding_model(50, classification = True)
gen = generate_batch(pairs, n_positive, negative_ratio=2, classification = True)

# 분류모델 학습
h = model.fit_generator(gen, epochs = 15, steps_per_epoch = len(pairs) // n_positive, verbose=1)

model_class.save('first_attempt_class.h5')

book_weights_class = extract_weights('book_embedding', model_class)
book_weights_class.shape

find_similar('The Better Angels of Our Nature', book_weights_class, n = 5, plot=True)


## visualization
from sklearn.manifold import TSNE

def reduce_dim(weights, components = 3, method = 'tsne'):
    """Reduce dimensions of embeddings"""
    method == 'tsne'
    return TSNE(components, metric = 'cosine').fit_transform(weights)

book_r = reduce_dim(book_weights_class, components = 2, method = 'tsne')
book_r.shape

InteractiveShell.ast_node_interactivity = 'last'

plt.figure(figsize = (10, 8))
plt.plot(book_r[:, 0], book_r[:, 1], 'r.')
plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('Book Embeddings Visualized with TSNE');
plt.show()


## book Embeddings by Genre
# book Genre 가져오기
info = list(chain(*[set(book[1]) for book in books ]))
# genre 개수 세기
info_counts = count_items(info)
# 상위 10개 뽑아오기
list(info_counts.items())[:10]


# ('genre', 'None') 값 찾아오기
genres = [book[1].get('genre','None').lower() for book in books]

#('genre', 'None') 개수 세기
genre_counts = count_items(genres)

# none값 제거하기
del genre_counts['none']

# genre 상위 10개 가져오기
list(genre_counts.iteam())[:10]


# Include 10 most popular genres
genre_to_include = list(genre_counts.keys())[:10]

idx_include = []
genres = []

for i, book in enumerate(books):
    if 'genre' in book[1].keys():
        if book[1]['genre'].lower() in genre_to_include:
            idx_include.append(i)
            genres.append(book[1]['genre'].capitalize())
            
len(idx_include)

ints, gen = pd.factorize(genres)
gen[:5]

plt.figure(figsize = (10, 8))

# Plot embedding
plt.scatter(book_r[idx_include, 0], book_r[idx_include, 1], 
            c = ints, cmap = plt.cm.tab10)

# Add colorbar and appropriate labels
cbar = plt.colorbar()
cbar.set_ticks([])
for j, lab in enumerate(gen):
    cbar.ax.text(1, (2 * j + 1) / ((10) * 2), lab, ha='left', va='center')
cbar.ax.set_title('Genre', loc = 'left')


plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('TSNE Visualization of Book Embeddings');
plt.show()