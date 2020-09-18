import os
import jieba
import re
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from tools import fetch_file_path
from corpus import knowledge_graph_root_path


class QuestionClassify:
    def __init__(self):
        self.tf_v = None
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(os.path.join(knowledge_graph_root_path, 'bayes.model')):
            self.model = joblib.load(os.path.join(knowledge_graph_root_path, 'bayes.model'))
            self.tf_v = joblib.load(os.path.join(knowledge_graph_root_path, 'tf_v.model'))
        else:
            self.train()

    def save_model(self):
        joblib.dump(self.tf_v, os.path.join(knowledge_graph_root_path, 'tf_v.model'))
        joblib.dump(self.model, os.path.join(knowledge_graph_root_path, 'bayes.model'))

    def read_data(self):
        x, y = [], []
        file_path = os.path.join(knowledge_graph_root_path, 'question')
        for one_file in [os.path.join(file_path, file) for file in fetch_file_path(file_path)]:
            num = re.sub(r'\D', '', one_file)
            if num.strip() != '':
                with open(one_file, mode='r', encoding='utf8') as fr:
                    for line in fr.readlines():
                        x.append(' '.join(jieba.cut(line.strip())))
                        y.append(int(num))
        return x, y

    def train(self):
        x, y = self.read_data()
        self.tf_v = TfidfVectorizer()
        self.tf_v.fit(x)
        train_data = self.tf_v.transform(x).toarray()
        self.model = MultinomialNB(alpha=0.01)
        self.model.fit(train_data, y)
        self.save_model()

    def predict(self, question):
        question = [' '.join(jieba.cut(question.strip()))]
        test_data = self.tf_v.transform(question).toarray()
        y_pred = self.model.predict(test_data)[0]
        return y_pred


if __name__ == '__main__':
    q = QuestionClassify()
    q.predict('刘德华的电影有哪些')
