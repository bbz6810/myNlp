import os

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, KFold

from corpus import news_classifier_path, news_test_path, tianchi_news_class_path
from tools import running_of_time

train_split = 160000


@running_of_time
def running():
    data_df = pd.read_csv(news_classifier_path, sep='\t')
    test_df = pd.read_csv(news_test_path, sep='\t')
    data_df['label_ft'] = '__label__' + data_df['label'].astype(str)
    print('总数据信息', data_df.info())

    location = np.random.permutation(data_df.shape[0])
    train_df = pd.DataFrame(data=data_df.values[location][:train_split], columns=['label', 'text', 'label_ft'])
    print('训练集数据信息', train_df.info())
    valid_df = pd.DataFrame(data=data_df.values[location][train_split:], columns=['label', 'text', 'label_ft'])
    print('验证集数据信息', valid_df.info())

    train_x, train_y = train_df['text'], train_df['label']
    valid_x, valid_y = valid_df['text'], valid_df['label']
    test_x = test_df['text']

    kf = KFold(n_splits=10, random_state=2020, shuffle=True)

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        train_df[['text', 'label_ft']].iloc[train_index].to_csv(
            os.path.join(tianchi_news_class_path, 'fasttext_train_df.csv'), header=None, index=False, sep='\t')
        model = fasttext.train_supervised(os.path.join(tianchi_news_class_path, 'fasttext_train_df.csv'), lr=0.1,
                                          epoch=27, wordNgrams=5, verbose=2, minCount=1, loss='hs')
        model.save_model(os.path.join(tianchi_news_class_path, 'fasttext_k_{}.model'.format(i)))

        model.get_sentence_vector()


if __name__ == '__main__':
    running()
