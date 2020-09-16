import os
import jieba
from corpus import knowledge_graph_root_path


class Question:
    def __init__(self):
        # user_dict classifier db
        self._init_config()

    def _init_config(self):
        # 分类器
        self.question_classifier = None
        # 问题模板
        self.question_template = self._read_question_template()
        # 创建问题模板对象
        self.question_template_obj = None

    def _read_question_template(self):
        d = dict()
        with open(os.path.join(knowledge_graph_root_path, 'question', 'question_classification.txt'), mode='r',
                  encoding='utf8') as f:
            for line in f.readlines():
                mode_id, mode_str = line.strip().split(':')
                d[int(mode_id)] = mode_str.strip()
        return d

    def process_question(self):
        pass
