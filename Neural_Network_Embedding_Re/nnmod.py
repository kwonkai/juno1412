import tensorflow as tf
import json
from itertools import chain
from collections import Counter, OrderedDict

class data_cleaning:
    def bookdata():
        x = tf.keras.utils.get_file('found_books_filtered.ndjson', 'https://raw.githubusercontent.com/WillKoehrsen/wikipedia-data-science/master/data/found_books_filtered.ndjson')
        books = []
        with open(x, 'r') as fin: # 'r' 읽기용으로 파일 열기
            books = [json.loads(l) for l in fin]
        
        return books
    
    def book_clean(books):
        books = [book for book in books if 'Wikipedia:' not in book[0]]
        book_index = {book[0] : idx for idx, book in enumerate(books)}
        index_book = {idx : book for book, idx in book_index.items()}

        return book_index, index_book

    def wikidata(bookdata):
        