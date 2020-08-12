"""BM25
    1、目的：计算query和doc的相关性得分 Score(Q, d) = ∑ W_i * R(q_i, d)
    2、query分词
    3、计算每个词的权重W_i(利用IDF计算)
        IDF(q_i) = log((N-n(q_i)+0.5) / (n(q_i)+0.5))
        N: 文档个数
        n(q_i): 包含了q_i词的文档个数
        由公式可以看出文档包含q_i词的个数越多W_i的权重值越低

    4、计算每个词和文档和相关性得分: R(q_i, d)
        一般形式:
            R(q_i, d) = [(f_i*(k_1 + 1)) / (f_i + K)] * [(qf_i*(k_2 + 1)) / (qf_i + k_2)]
            K = k_1 * (1 - b + b * deep_learning / avgdl)

            其中k_1, k_2, b 是调节因子，可根据实际情况设定，一般 k_1 = 2, b = 0.75
            1. f_i 为 q_i 在文档d中出现的频率
            2. qf_i 为 q_i 在query中出现的频率
            3. deep_learning 为当前query的长度， avgdl 为所有文档的平均长度

            由于大部分情况下 qf_i 都为 1，所以上面的Score式子可以简化为：

            R(q_i, d) = (f_i*(k_1+1)) / (f_i + K)
                      = (f_i * (k_1 + 1)) / (f_i + (k_1 * (1-b+b*deep_learning/avgdl)))
            注：
                从K的定义可以看到参数b的作用是调节文档长度对相关性的影响。
                b越大则文档长度对相关性的影响越大，反之越小
                文档的相对长度越大 K 越大，则相关性得分越小。
                这可以理解为当文档较长时，包含q_i的机会越大，
                同等情况下长文档与q_i的相关性比段文档与q_i
                的相关性弱

    5、综上Score公式为：
        Score(query, d) = ∑ IDF(q_i) * [(f_i * (k_1 + 1)) / (f_i + (k_1 * (1-b+b*deep_learning/avgdl)))]
"""
import jieba

from tools import filter_stop
from model.tf_idf import TFIDF


class BM25:
    def __init__(self):
        self.tf_idf = TFIDF()
        self.k_1 = 2
        self.k_2 = 2
        self.b = 0.75

    def load(self):
        self.tf_idf.load('/projects/myNlp/textCategory/20200711')

    def train(self):
        pass

    def get_idf(self, word):
        word_index = self.tf_idf.word_index.get(word)
        if word_index:
            return self.tf_idf.idf[word_index]
        else:
            print('get_idf:没有该词:{}'.format(word))
            return 1

    def get_f_i(self, word, doc_index):
        word_index = self.tf_idf.word_index.get(word)
        if word_index:
            return self.tf_idf.tf_list[doc_index][word_index]
        else:
            print('get_f_i:没有该词:{}'.format(word))
            return 1

    def word_score(self, word, doc_index, K):
        # ∑ IDF(q_i) * [(f_i * (k_1 + 1)) / (f_i + (k_1 * (1-b+b*deep_learning/avgdl)))]
        idf = self.get_idf(word)
        f_i = self.get_f_i(word, doc_index)
        return idf * ((f_i * (self.k_1 + 1)) / (f_i + K))

    def get_dl_avgdl(self, word):
        total = sum(sum(i) for i in self.tf_idf.np_tf_list)
        return len(word), total / self.tf_idf.np_tf_list.shape[0]

    def get_K(self, dl, avgdl):
        return self.k_1 + self.b - self.b * (dl / avgdl)

    def get_doc_word(self, doc_index):
        word_index = {v: k for k, v in self.tf_idf.word_index.items()}
        for index, i in enumerate(self.tf_idf.np_tf_list[doc_index]):
            if i > 0:
                print(word_index[index], end=' ')

    def predict(self, key):
        self.load()
        key = filter_stop(jieba.cut(key))
        dl, avgdl = self.get_dl_avgdl(key)
        K = self.get_K(dl, avgdl)
        max_doc_index, max_score = 0, 0
        for doc_index in range(self.tf_idf.tf_list.shape[0]):
            score = 0
            for k in key:
                one_word_score = self.word_score(k, doc_index, K)
                score += one_word_score
            if score > max_score:
                max_score, max_doc_index = score, doc_index
        print('max', max_doc_index, max_score)
        self.get_doc_word(max_doc_index)


if __name__ == '__main__':
    bm25 = BM25()
    bm25.predict('据悉,WWEC教育者大会是由君学发起、上海雁传书文化传媒承办的,聚集国内外资本、学术、教育、文艺、政商等多领域的专家、学者、社会精英人士共同参与的全球教育行业跨界交流盛会,经过7年的发展沉淀,已逐渐成为泛教育产业的一个标志性符号。大会自2013年8月20日在上海国际会议中心召开第一届“素质教育”以来,分别以“科技教育”、“生命教育”、“教育+”、“公益教育”、“未来教育”、“以热爱拥抱未来”为中心思想,成功举办了七届,吸引了全球20多个国家、数万家教育机构共同参与,辐射和影响到的社会各界人群多达数千万。7年来的积累和沉淀,赢得了社会各界中外人士的广泛赞誉。')
