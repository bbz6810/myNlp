import os

corpus_root_path = '/Users/zhoubb/projects/corpus'

news_jieba_path = os.path.join(corpus_root_path, 'news_fasttext_train_jieba.txt')

wechat_new_word_data_path = os.path.join(corpus_root_path, 'wechat_new_word_data.txt')

paper_path = os.path.join(corpus_root_path, 'people1998.txt')
paper_wv_path = os.path.join(corpus_root_path, 'sgns.renmin.bigram-char')

line_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'line_r.txt')

category_path = {
    'edu': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'C5-Education'),
    'phy': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'C6-Philosophy')
}

imdb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imdb.npz')
imdb_word_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imdb_word_index.json')

wv_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'wv_model.model')

stop_path = os.path.join(corpus_root_path, 'stopwords.txt')
