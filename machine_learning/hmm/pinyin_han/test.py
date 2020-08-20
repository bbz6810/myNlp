"""mai xiang chong man
构建二元语法
"""
import math

from pypinyin import lazy_pinyin
from corpus.load_corpus import LoadCorpus
from tools import running_of_time

pinyin_han_dict = dict()
start_dict = dict()
trans_dict = dict()
emit_dict = dict()

MIN_FLOAT = -3.14e100


@running_of_time
def pre_data():
    word_count = {}
    # sentences = [''.join(sentence) for sentence in LoadCorpus.load_paper_to_word2vec()]
    sentences, _ = LoadCorpus.load_news_train(c=1)
    sentences = [sentence.replace(' ', '') for sentence in sentences]

    for sentence in sentences:
        for char in sentence:
            if char not in word_count:
                word_count[char] = 0
            word_count[char] += 1

    word_total = sum(word_count.values())
    for word, value in word_count.items():
        start_dict[word] = math.log(value / word_total)

    han_pinyin_dict = dict((han, py) for han, py in zip(word_count.keys(), lazy_pinyin(list(word_count.keys()))))
    sentences = [[sentence[i:i + 2] for i in range(len(sentence) - 1)] for sentence in sentences]

    for sentence in sentences:
        for tup in sentence:
            if tup[0] not in trans_dict:
                trans_dict[tup[0]] = {}
            if tup[1] not in trans_dict[tup[0]]:
                trans_dict[tup[0]][tup[1]] = 0
            trans_dict[tup[0]][tup[1]] += 1

    for han, pinyin in han_pinyin_dict.items():
        if pinyin not in pinyin_han_dict:
            pinyin_han_dict[pinyin] = []
        pinyin_han_dict[pinyin].append(han)

    for key, value in pinyin_han_dict.items():
        total = sum(word_count[v] for v in value)
        for v in value:
            if v not in emit_dict:
                emit_dict[v] = dict()
            emit_dict[v][key] = math.log(word_count[v] / total)


def get_status(pinyin):
    return pinyin_han_dict[pinyin]


@running_of_time
def viterbi(sentence):
    if not sentence:
        return ''
    v = [{}]
    path = {}
    status = []
    pin_yin_list = sentence.split()
    for index, pin_yin in enumerate(pin_yin_list):
        up_status = status
        status = get_status(pin_yin)
        new_path = {}
        if index == 0:
            for han in status:
                v[index][han] = start_dict[han] + emit_dict[han][pin_yin]
                path[han] = [han]
        else:
            for han in status:
                v.append({})
                emit = emit_dict[han][pin_yin]
                prob, state = max(
                    [(v[index - 1].get(han0, MIN_FLOAT) + emit + trans_dict.get(han0, {}).get(han, MIN_FLOAT), han0) for
                     han0 in up_status])
                v[index][han] = prob
                new_path[han] = path.get(state, []) + [han]
            path = new_path
    prob, state = max([v[len(pin_yin_list) - 1][han], han] for han in get_status(pin_yin_list[-1]))
    print(prob, path[state])


if __name__ == '__main__':
    pre_data()
    # a = lazy_pinyin('迈向充满希望的新世纪--一九九八年新年讲话附图片１张')
    # print(a)
    viterbi('jiao yu shi min zu de xi wang')
    # sentences = [''.join(sentence) for sentence in LoadCorpus.load_paper_to_word2vec()]
    # for sentence in sentences:
    #     print(sentence)
    #     try:
    #         viterbi(' '.join(lazy_pinyin(sentence)))
    #     except Exception as e:
    #         pass
