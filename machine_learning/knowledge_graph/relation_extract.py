import jieba
from jieba import posseg
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SentenceSplitter
from corpus import cws_model, pos_model, parser_model, ner_model, news_path, relation_extract_output_file

segmentor = Segmentor()
segmentor.load(cws_model)

postagger = Postagger()
postagger.load(pos_model)

parser = Parser()
parser.load(parser_model)

recognizer = NamedEntityRecognizer()
recognizer.load(ner_model)

sentencesplit = SentenceSplitter()


def extract_start(input_file_name, output_file_name, begin_line, end_line):
    in_file = open(input_file_name, 'r', encoding='utf8')
    out_file = open(output_file_name, 'w')

    for line in in_file.readlines()[begin_line:end_line]:
        for sentence in sentencesplit.split(''.join(line.split()[:-1])):
            fact_extract(sentence, out_file)
    # fact_extract('欧几里得是西元前三世纪的希腊数学家。', out_file)

    in_file.close()
    out_file.close()


def fact_extract(sentence, out_file):
    # print('sentence', sentence)
    # words = segmentor.segment(sentence)
    words = jieba.lcut(sentence)
    # for word in words:
    #     print(word, end=' ')
    # print()
    postags = postagger.postag(words)
    # for pos in postags:
    #     print(pos, end=' ')
    # print()
    netage = recognizer.recognize(words, postags)
    # for net in netage:
    #     print(net, end=' ')
    # print()
    arcs = parser.parse(words, netage)

    relay_id = [arc.head for arc in arcs]
    # print(relay_id)
    relation = [arc.relation for arc in arcs]
    # print(relation)
    heads = ['root' if idx == 0 else words[idx - 1] for idx in relay_id]
    # print(heads)
    # for i in range(len(words)):
    #     print(words[i], heads[i])
    child_dict_list = build_child_dict(words, postags, arcs)
    # print(child_dict_list)
    for index in range(len(postags)):
        if postags[index] == 'v':
            child_dict = child_dict_list[index]
            if child_dict.get('SBV') and child_dict.get('VOB'):
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                r = words[index]
                e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                print('主谓宾关系:{},{},{}'.format(e1, r, e2))

            if arcs[index].relation == 'ATT':
                if child_dict.get('VOB'):
                    e1 = complete_e(words, postags, child_dict_list, arcs[index].head - 1)
                    r = words[index]
                    e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    temp_string = r + e2
                    if temp_string == e1[:len(temp_string)]:
                        e1 = e1[len(temp_string):]
                    if temp_string not in e1:
                        print('定语后置关系:{},{},{}'.format(e1, r, e2))

            if child_dict.get('SBV') and child_dict.get('CMP'):
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                cmp_index = child_dict['CMP'][0]
                r = words[index] + words[cmp_index]
                if child_dict_list[cmp_index].get('POB'):
                    e2 = complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                    print('介宾关系主谓动补:{},{},{}'.format(e1, r, e2))

        if netage[index][0] == 'S' or netage[index][0] == 'B':
            ni = index
            if netage[ni][0] == 'B':
                while netage[ni][0] != 'E':
                    ni += 1
                e1 = ''.join(words[index:ni + 1])
            else:
                e1 = words[ni]
            if arcs[ni].relation == 'ATT' and postags[arcs[ni].head - 1] == 'n' and netage[arcs[ni].head - 1] == 'O':
                r = complete_e(words, postags, child_dict_list, arcs[ni].head - 1)
                if e1 in r:
                    r = r[(r.index(e1) + len(e1)):]
                if arcs[arcs[ni].head - 1].relation == 'ATT' and netage[arcs[arcs[ni].head - 1].head - 1] != 'O':
                    e2 = complete_e(words, postags, child_dict_list, arcs[arcs[ni].head - 1].head - 1)
                    mi = arcs[arcs[ni].head - 1].head - 1
                    li = mi
                    if netage[mi][0] == 'B':
                        while netage[mi][0] != 'E':
                            mi += 1
                        e = ''.join(words[li:mi + 1])
                        e2 += e
                    if r in e2:
                        e2 = e2[(e2.index(r) + len(r)):]
                    if r + e2 in sentence:
                        print('人名，地名，机构名:'.format(e1, r, e2))


def build_child_dict(words, postags, arcs):
    child_dict_list = list()
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:
                if arcs[arc_index].relation not in child_dict:
                    child_dict[arcs[arc_index].relation] = []
                child_dict[arcs[arc_index].relation].append(arc_index)
        child_dict_list.append(child_dict)
    return child_dict_list


def complete_e(words, postags, child_dict_list, word_index):
    child_dict = child_dict_list[word_index]
    prefix = ''
    if child_dict.get('ATT'):
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
    postfix = ''
    if postags[word_index] == 'v':
        if child_dict.get('VOB'):
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if child_dict.get('SBV'):
            prefix += complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix
    return prefix + words[word_index] + postfix


if __name__ == '__main__':
    extract_start(news_path, relation_extract_output_file, 0, 10)
