"""
两部分：1、模型训练。2、分词
训练模型：

"""
import json
import math
import re

from machine_learning.hmm.split import P as trans
from machine_learning.hmm.split.start import P as start
from machine_learning.hmm.split import P as emit

source_text = 'D:\code\coupurs\cut\paper_199801.txt'

start_status_file_name = 'start.txt'
trans_status_file_name = 'trans.txt'
emit_status_file_name = 'emit.txt'

dict_txt = 'd:/code/myNlp/split/myJieba/dict1.txt'

MIN_FLOAT = -3.14e100

PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}


class myJieba:
    def __init__(self):
        # 观测序列

        # 隐藏状态
        self.init_status = 'BMES'
        # 状态开始矩阵
        self.start_status = {}

        # 状态转移矩阵
        self.trans_status = {}

        # 发射矩阵
        self.emit_status = {}

        # 前缀字典
        self.freq = {}
        self.total = 0

        # self.load_model_file()
        self.build_model()

    def _load_file(self, file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    def _dump_file(self, file_name, _dict):
        with open(file_name, 'w') as f:
            return json.dump(_dict, f)

    def load_model_file(self):
        # self.start_status = self._load_file(start_status_file_name)
        # self.trans_status = self._load_file(trans_status_file_name)
        # self.emit_status = self._load_file(emit_status_file_name)

        self.start_status = start
        self.trans_status = trans
        self.emit_status = emit

    def dump_model_file(self):
        self._dump_file(start_status_file_name, self.start_status)
        self._dump_file(trans_status_file_name, self.trans_status)
        self._dump_file(emit_status_file_name, self.emit_status)

    def parse_dict_txt(self):
        _start_status = {'B': 1, 'M': 1, 'E': 1, 'S': 1}
        len_words = 0
        with open(dict_txt, 'r', encoding='utf8') as f:
            for line in f.readlines():
                word, _, _ = line.strip().split(' ')
                len_words += len(word)
                if len(word) == 1:
                    _start_status['S'] += 1
                elif len(word) > 1:
                    _start_status['B'] += 1
        for i in _start_status:
            _start_status[i] = math.log2(_start_status[i] / len_words)

    def check_init(self):
        self.freq, self.total = self.get_pf()

    def get_pf(self):
        """ 根据词典生成前缀词典

        :return:
        """
        lfreq = {}
        ltotal = 0
        with open(dict_txt, 'r', encoding='utf8') as f:
            for line in f.readlines():
                word, freq = line.strip().split(' ')[:2]
                lfreq[word] = int(freq)
                ltotal += int(freq)
                for i, w in enumerate(word):
                    if word[:i + 1] not in lfreq:
                        lfreq[word[:i + 1]] = 0
        return lfreq, ltotal

    def get_DAG(self, sentence):
        """生成有向无环图

        :return:
        """
        self.check_init()
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            freq = sentence[k]
            while i < N and freq in self.freq:
                if self.freq.get(freq):
                    tmplist.append(i)
                i += 1
                freq = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def calc(self, sentence, DAG, route):
        """从后向前计算最大概率路径(每到一个节点，该节点前一个节点到最后的最大路径概率已经计算出来)

        :param sentence:
        :param DAG:
        :param route:
        :return:
        """
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = math.log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max(
                (math.log(self.freq[sentence[idx:i + 1]] or 1) - logtotal + route[i + 1][0], i) for i in DAG[idx])
        return route

    def cut_no_hmm(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
            x = y
            if buf:
                yield buf
                buf = ''

    def cut(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not self.freq.get(buf):
                            # HMM
                            for c in self.cut_hmm(buf):
                                yield c
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            else:
                if not self.freq.get(buf):
                    # HMM
                    for c in self.cut_hmm(buf):
                        yield c
                else:
                    for elem in buf:
                        yield elem

    def cut_hmm(self, sentence):
        re_han = re.compile("([\u4E00-\u9FD5]+)")
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                for c in self._cut_hmm(blk):
                    yield c

    def _cut_hmm(self, sentence):
        prob, states = self.viterbi(sentence, 'BMES')
        begin, nexti = 0, 0
        for i, char in enumerate(sentence):
            if states[i] == 'B':
                begin = i
            elif states[i] == 'E':
                yield sentence[begin:i + 1]
                nexti = i + 1
            elif states[i] == 'S':
                yield char
                nexti = i + 1
        if nexti < len(sentence):
            yield sentence[nexti:]

    def viterbi(self, sentence, status):
        V = [{}]
        path = {}
        for y in status:
            V[0][y] = self.start_status[y] + self.emit_status[y].get(sentence[0], MIN_FLOAT)
            path[y] = [y]
        for t in range(1, len(sentence)):
            V.append({})
            newpath = {}
            for y in status:
                emit = self.emit_status[y].get(sentence[t], MIN_FLOAT)
                prob, state = max(
                    [(V[t - 1][y0] + self.trans_status[y0].get(y, MIN_FLOAT) + emit, y0) for y0 in PrevStatus[y]])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        prob, state = max((V[len(sentence) - 1][y], y) for y in 'ES')
        return prob, path[state]

    def __str__(self):
        return "start_status: {}\ntrans_status: {}\nemit_status: {}".format(
            self.start_status, self.trans_status, self.emit_status)

    def build_model(self):
        # 隐藏状态初始化矩阵
        _start = dict()
        # 状态转移矩阵
        _trans = dict()
        # 发射矩阵
        _emit = dict()

        def word_format(w):
            if len(w) == 1:
                return 'S'
            retv_str = ''
            for i, k in enumerate(w):
                if i == 0:
                    retv_str += 'B'
                elif i == len(w) - 1:
                    retv_str += 'E'
                else:
                    retv_str += 'M'
            return retv_str

        word_dict = dict()

        with open(source_text, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line_state = ''
                line_word = ''
                for word in line.strip().split('  ')[1:]:
                    # 状态初始化矩阵
                    words = word.split('/')[0]
                    if word_dict.get(words):
                        word_dict[words] += 1
                    else:
                        word_dict[words] = 1
                    if len(words) == 1:
                        if 'S' in _start:
                            _start['S'] += 1
                        else:
                            _start['S'] = 1
                    else:
                        if 'B' in _start:
                            _start['B'] += 1
                        else:
                            _start['B'] = 1
                    # 状态转移矩阵
                    line_state += word_format(words)
                    line_word += words

                for i in range(len(line_state) - 1):
                    if _trans.get(line_state[i]):
                        if _trans[line_state[i]].get(line_state[i + 1]):
                            _trans[line_state[i]][line_state[i + 1]] += 1
                        else:
                            _trans[line_state[i]][line_state[i + 1]] = 1
                    else:
                        _trans[line_state[i]] = {line_state[i + 1]: 1}

                for i in range(len(line_state)):
                    if _emit.get(line_state[i]):
                        if _emit[line_state[i]].get(line_word[i]):
                            _emit[line_state[i]][line_word[i]] += 1
                        else:
                            _emit[line_state[i]][line_word[i]] = 1
                    else:
                        _emit[line_state[i]] = {line_word[i]: 1}

        # print('_start up', _start)
        d = sum(_start.values())
        for key in 'BMES':
            if _start.get(key):
                _start[key] = math.log(_start[key] / d, math.e)
            else:
                _start[key] = MIN_FLOAT
        # print('_start down', _start)

        # 状态转移矩阵
        # print('_trans up', _trans)
        d = {k: sum(v.values()) for k, v in _trans.items()}
        for k, v in _trans.items():
            for kk, vv in v.items():
                _trans[k][kk] = math.log(vv / d[k], math.e)
        # print('_trans down', _trans)

        # 发射矩阵
        # print('_emit up', _emit)
        d = {i: sum(v.values()) for i, v in _emit.items()}
        for k, v in _emit.items():
            for kk, vv in v.items():
                _emit[k][kk] = math.log(vv / d[k], math.e)
        # print('_emit down', _emit)

        self.start_status = _start
        self.trans_status = _trans
        self.emit_status = _emit
        # print(word_dict)
        # with open('dict1.txt', encoding='utf8', mode='w') as f:
        #     for k, v in word_dict.items():
        #         f.write('{} {}\n'.format(k, v))


if __name__ == '__main__':
    # myJieba().build_model()
    # words = '今天天气很好'
    words = '祖国的大好河山就在我们脚下'
    model = myJieba()
    # model.viterbi('今天天气很好', 'BMES')
    l = model.cut(words)
    print(list(l))
    # print(jieba.lcut(words))
