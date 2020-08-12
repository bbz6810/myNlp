from corpus.load_corpus import LoadCorpus


def test():
    x, y = LoadCorpus.load_chatbot100_train()
    wv = LoadCorpus.load_wv_model()
    # print(x)
    # print(y)
    print(len(wv.wv.vectors))


if __name__ == '__main__':
    test()
