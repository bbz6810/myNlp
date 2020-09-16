"""
对话-翻译-语料例子：http://www.manythings.org/anki/
"""

import os

corpus_root_path = '/Users/zhoubb/projects/corpus'

news_jieba_path = os.path.join(corpus_root_path, 'news_fasttext_train_jieba.txt')
news_path = os.path.join(corpus_root_path, 'news_fasttext_train.txt')
chatbot100_path = os.path.join(corpus_root_path, 'chatbot100')
xiaohuangji_path = os.path.join(corpus_root_path, 'xiaohuangji50w_fenciA.conv')
seq2seq_model_path = os.path.join(corpus_root_path, 'seq2seq.h5')
seq2seq2_model_path = os.path.join(corpus_root_path, 'seq2seq2.h5')
char2char_obj_path = os.path.join(corpus_root_path, 'char2char_obj')
word2word_obj_path = os.path.join(corpus_root_path, 'word2word_obj')

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

chinese_to_english_path = os.path.join(corpus_root_path, 'cmn-eng/cmn.txt')
french_to_english_path = os.path.join(corpus_root_path, 'fra-eng/fra.txt')

crf_model_path = os.path.join(corpus_root_path, 'crf_model.h5')
word2vec_model_path = os.path.join(corpus_root_path, 'word2vec_model')

cat_data = os.path.join(corpus_root_path, 'car.data')

cws_model = os.path.join(corpus_root_path, 'ltp_data', "cws.model")
pos_model = os.path.join(corpus_root_path, 'ltp_data', "pos.model")
parser_model = os.path.join(corpus_root_path, 'ltp_data', "parser.model")
ner_model = os.path.join(corpus_root_path, 'ltp_data', "ner.model")

relation_extract_output_file = os.path.join(corpus_root_path, 'relation_extract_output.txt')

# 天池比赛数据
news_classifier_path = os.path.join(corpus_root_path, 'train_set.csv')
news_test_path = os.path.join(corpus_root_path, 'test_b.csv')
news_one_to_one_path = os.path.join(corpus_root_path, 'one2one')
daikuan_path = os.path.join(corpus_root_path, 'daikuan')
daikuan_classifier_path = os.path.join(daikuan_path, 'train.csv')
daikuan_test_path = os.path.join(daikuan_path, 'testA.csv')

# applications data
knowledge_graph_root_path = os.path.join(corpus_root_path, 'question_answer_data')
