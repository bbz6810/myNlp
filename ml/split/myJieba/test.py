import math

dict_txt = 'd:/code/myNlp/split/myJieba/dict.txt'


def my_jieba(sentence):
    # 构建前缀词典
    freq_dict = {}
    freq_total = 0
    with open(dict_txt, 'r', encoding='utf8') as f:
        for line in f.readlines():
            word, freq, _ = line.strip().split(' ')
            freq_dict[word] = int(freq)
            freq_total += int(freq)
            for i, w in enumerate(word):
                if not freq_dict.get(word[:i + 1]):
                    freq_dict[word[:i + 1]] = 0
    # 生成DAG有向无环图
    DAG = dict()
    N = len(sentence)
    for i in range(N):
        tmplist = []
        j = i
        freq = sentence[i]
        while j < N and freq in freq_dict:
            if freq_dict.get(freq):
                tmplist.append(j)
            j += 1
            freq = sentence[i:j + 1]
        if not tmplist:
            tmplist.append(i)
        DAG[i] = tmplist

    print(DAG)

    # 根据有向无环图计算最大路径概率
    route = dict()
    route[N] = (0, 0)
    logtotal = math.log(freq_total)
    for idx in range(N - 1, -1, -1):
        route[idx] = max(
            (math.log(freq_dict.get(sentence[idx:i + 1])) - logtotal + route[i + 1][0], i) for i in DAG[idx])
    print(route)
    x = 0
    while x < N:
        print(x, route[x][1]+1, sentence[x:route[x][1] + 1])
        x = route[x][1] + 1
        print(x)



if __name__ == '__main__':
    my_jieba('我爱天天向上')
