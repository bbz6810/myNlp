import os
import re
import jieba
from jieba import posseg
from applications.knowledge_graph.question_classifier import QuestionClassify
from applications.knowledge_graph.question_template import QuestionTemplate
from corpus import knowledge_graph_root_path


class Question:
    def __init__(self):
        # user_dict classifier db
        self._init_config()
        self.question_word = None
        self.question_flag = None
        self.question_template_id_str = None

    def _init_config(self):
        # 分类器
        self.question_classifier = QuestionClassify()
        # 问题模板
        self.question_template = self._read_question_template()
        # 创建问题模板对象
        self.question_template_obj = QuestionTemplate()

    def _read_question_template(self):
        d = dict()
        with open(os.path.join(knowledge_graph_root_path, 'question', 'question_classification.txt'), mode='r',
                  encoding='utf8') as f:
            for line in f.readlines():
                mode_id, mode_str = line.strip().split(':')
                d[int(mode_id)] = mode_str.strip()
        return d

    def process_question(self, question):
        result, self.question_word, self.question_flag = self.posseg_question(question.strip())
        print(result)
        self.question_template_id_str = self.fetch_question_template()
        print(self.question_template_id_str)
        answer = self.query_template(result)
        return answer

    def posseg_question(self, question):
        result = []
        question_word = []
        question_flag = []
        jieba.load_userdict(os.path.join(knowledge_graph_root_path, 'userdict3.txt'))
        question = re.sub('[\s+.!/_,$%^*(\"\')]+|[+—()?【】“”！，。？、~@#￥%…&*（）]+', '', question)
        for w in posseg.cut(question):
            result.append('{}/{}'.format(w.word, w.flag))
            question_word.append(w.word.strip())
            question_flag.append(w.flag.strip())
        return result, question_word, question_flag

    def fetch_question_template(self):
        for item in ['nr', 'nm', 'ng']:
            while item in self.question_flag:
                idx = self.question_flag.index(item)
                self.question_word[idx] = item
                self.question_flag[idx] = item + 'ed'
        question_template_num = self.question_classifier.predict(''.join(self.question_word))
        question_template = self.question_template[question_template_num]
        question_template_id_str = '{}\t{}'.format(question_template_num, question_template)
        return question_template_id_str

    def query_template(self, result):
        try:
            answer = self.question_template_obj.get_question_answer(result, self.question_template_id_str)
        except:
            answer = '我也不知道！'
        return answer


if __name__ == '__main__':
    q = Question()
    r = q.process_question('成龙参演的电影有多少部')
    print(r)
