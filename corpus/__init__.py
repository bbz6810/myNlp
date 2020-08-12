import os

wechat_new_word_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wechat_new_word_data.txt')

paper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'people1998.txt')

line_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'line_r.txt')

category_path = {
    'edu': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'C5-Education'),
    'phy': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'C6-Philosophy')
}

imdb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imdb.npz')
imdb_word_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imdb_word_index.json')

wv_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'wv_model.model')

stop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt')
