import re

from ml.split.myJieba import myJieba

from ml.pos.HMM import P as state_tab
from ml.pos.HMM.start import P as start
from ml.pos.HMM import P as trans
from ml.pos.HMM.emit import P as emit

re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile("(\r\n|\s)")

re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile("[\.0-9]+")

re_eng1 = re.compile('^[a-zA-Z0-9]$', re.U)

word_tag_filename = 'd:/code/myNlp/split/myJieba/dict.txt'
MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")


class pair:
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __str__(self):
        return 'pair:[{} /{}]'.format(self.word, self.flag)

    def __unicode__(self):
        return 'pair:[{} /{}]'.format(self.word, self.flag)


class PosToken:
    def __init__(self):
        self.split = myJieba()
        self.word_tag_table = {}
        self.load_word_tag()

    def load_word_tag(self):
        with open(word_tag_filename, encoding='utf8') as f:
            for lineno, i in enumerate(f.readlines()):
                word, _, tag = i.strip().split(' ')
                self.word_tag_table[word] = tag

    def cut(self, sentence, HMM=True):
        for blk in self.__cut_internal(sentence, False):
            yield blk

    def __cut_internal(self, sentence, HMM=True):
        blocks = re_han_internal.split(sentence)
        if HMM is True:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM
        for blk in blocks:
            if re_han_internal.match(blk):
                for word in cut_blk(blk):
                    yield word

    def __cut_DAG(self, sentence):
        DAG = self.split.get_DAG(sentence)
        route = {}
        self.split.calc(sentence, DAG, route)
        x = 0
        buf = ''
        N = len(sentence)

        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield pair(buf, self.word_tag_table.get(buf, 'x'))
                    elif not self.split.freq.get(buf):
                        token = self.__cut_detail(buf)
                        for t in token:
                            yield t
                    else:
                        for elem in buf:
                            yield pair(elem, self.word_tag_table.get(elem, 'x'))
                    buf = ''
                yield pair(l_word, self.word_tag_table.get(l_word, 'x'))
            x = y

        if buf:
            if len(buf) == 1:
                yield pair(buf, self.word_tag_table.get(buf, 'x'))
            elif not self.split.freq.get(buf):
                token = self.__cut_detail(buf)
                for t in token:
                    yield t
            else:
                for elem in buf:
                    yield pair(elem, self.word_tag_table.get(elem, 'x'))

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.split.get_DAG(sentence)
        route = {}
        self.split.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng1.match(l_word):
                buf += l_word
            else:
                if buf:
                    yield pair(buf, 'eng')
                    buf = ''
                yield pair(l_word, self.word_tag_table.get(l_word, 'x'))
            x = y
        if buf:
            yield pair(buf, 'eng')

    def __cut_detail(self, sentence):
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):
                for word in self.__cut(blk):
                    yield word
            else:
                pass

    def __cut(self, sentence):
        prob, pos_list = self.viterbi(sentence, state_tab, start, trans, emit)
        print(prob, pos_list)
        begin, nexti = 0, 0
        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield pair(sentence[begin:i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == 'S':
                yield pair(sentence[i], pos_list[i][1])
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    def viterbi(self, sentence, states, start_p, trans_p, emit_p):
        V = [{}]
        mem_path = [{}]
        all_status = start_p.keys()
        for state in states.get(sentence[0], all_status):
            V[0][state] = start_p[state] + emit_p[state].get(sentence[0], MIN_FLOAT)
            mem_path[0][state] = ''
        for x in range(1, len(sentence)):
            V.append({})
            mem_path.append({})
            # 如果前一个状态的转移状态里有可转移的状态 则取此状态
            prev_states = [k for k in mem_path[x - 1].keys() if len(trans_p[k]) > 0]
            prev_states_expect_next = set([z for y in prev_states for z in trans_p[y].keys()])
            sentence_states = set(states.get(sentence[x], all_status)) & prev_states_expect_next

            if not sentence_states:
                sentence_states = prev_states_expect_next if prev_states_expect_next else all_status

            for y in sentence_states:
                prob, state = max((V[x-1][y0] + trans_p[y0].get(y, MIN_INF) +
                                   emit_p[y].get(sentence[x], MIN_FLOAT), y0) for y0 in prev_states)
                V[x][y] = prob
                mem_path[x][y] = state
        prob, state = max([(V[-1][y], y) for y in mem_path[-1].keys()])
        route = [None] * len(sentence)
        i = len(sentence) - 1
        while i >= 0:
            route[i] = state
            state = mem_path[i][state]
            i -= 1
        return prob, route


if __name__ == '__main__':
    p = PosToken()
    d = p.cut('中国话是世界上使用最多的人的语言')
    for i in d:
        print(i)
