import os

corpus_root_path = '/Users/zhoubb/projects/corpus'

news_jieba_path = os.path.join(corpus_root_path, 'news_fasttext_train_jieba.txt')
chatbot100_path = os.path.join(corpus_root_path, 'chatbot100')
xiaohuangji_path = os.path.join(corpus_root_path, 'xiaohuangji50w_fenciA.conv')
xiaohuangji_model_path = os.path.join(corpus_root_path, 'xiaohuangji_lstm.h5')

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

wv_model_path = os.path.join(corpus_root_path, 'sgns.renmin.bigram-char')

wv60_model_path = os.path.join(corpus_root_path, 'Word60.model')

stop_path = os.path.join(corpus_root_path, 'stopwords.txt')
