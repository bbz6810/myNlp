import os
import sys

sys.path.append('/Users/zhoubb/projects/myNlp')

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

from corpus import news_classifier_path, news_test_path, tianchi_news_class_path
from tools import running_of_time

np.random.seed(1)


@running_of_time
def build_word2vec():
    word2vec_name = os.path.join(tianchi_news_class_path, 'char2vec.wv')
    if not os.path.exists(word2vec_name):
        data_df = pd.read_csv(news_classifier_path, sep='\t')
        test_df = pd.read_csv(news_test_path, sep='\t')
        all_data = pd.concat([data_df['text'], test_df['text']])
        model = Word2Vec([[word for word in document.split()] for document in all_data.values],
                         size=200, window=5, iter=10, workers=3, seed=2, min_count=2)
        model.save(word2vec_name)
    else:
        model = Word2Vec.load(word2vec_name)
    return model


if __name__ == '__main__':
    build_word2vec()
