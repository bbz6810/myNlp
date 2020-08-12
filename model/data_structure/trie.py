"""trie树

"""
import time

word_path = '/projects/myNlp/ml/wechat_new_word_data.txt'


def read_data():
    d = {}
    with open(word_path, encoding='utf8', mode='r') as f:
        for line in f.readlines():
            t = line.strip().split(',')
            d[t[0]] = int(t[1])
    return d


class TrieDict(dict):
    def __init__(self, value=None):
        super(dict, self).__init__()
        self.value = value


d = TrieDict()


def create_trie():
    s = {'a': 1, 'ab': 2, 'ac': 3}
    s = read_data()
    for key, value in s.items():
        loop_build_trie(key, value)

    t = trie_find_key('饭')
    print(t)

    # now = time.time()
    # for i in range(100):
    #     for k in s:
    #         s.get(k)
    # print('字典', time.time() - now)
    #
    # now = time.time()
    # for i in range(100):
    #     for k in s:
    #         trie_find_key(k)
    # print('trie', time.time() - now)


def loop_build_trie(key, value):
    t = d
    for k in key:
        if k in t:
            t = t[k]
        else:
            t[k] = TrieDict(value)
            t = t[k]
    # print(d)


def trie_find_key(key):
    t = d
    value = None
    for k in key:
        if t.get(k):
            value = t[k].value
            t = t[k]
        else:
            return False, -1
    return True, value


if __name__ == '__main__':
    create_trie()
    # loop_build_trie('abc')
