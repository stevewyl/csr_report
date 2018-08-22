"""TFIDF计算和词向量训练"""

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim import corpora
import pandas as pd
from pathlib import Path
import pickle

WINDOWS_SIZE = 5
MIN_CNT = 10


def word2vec(text, fname, ndims, window_size, min_cnt=1):
    model = Word2Vec(text, min_count=min_cnt, window=window_size, size=ndims)
    model.wv.save_word2vec_format(fname, binary=False)
    return model


def to_dict(list_tuple, id2token):
    d = {id2token[item[0]]: item[1] for item in list_tuple}
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return d


def tfidf(text):
    # 生成字典和向量语料
    dictionary = Dictionary(text)
    id2token = {v: k for k, v in dictionary.token2id.items()}
    # 每一篇文档对应的稀疏向量
    # 向量的每一个元素代表了一个word在这篇文档中出现的次数
    corpus = [dictionary.doc2bow(d) for d in text]
    corpora.MmCorpus.serialize('./result/corpus.mm', corpus)
    tfidf_model = TfidfModel(corpus, id2word=dictionary)
    corpus_tfidf = tfidf_model[corpus]
    res = [to_dict(mat, id2token) for mat in corpus_tfidf]
    return res, id2token


if __name__ == "__main__":
    df = pd.read_csv('./result/all_csr_text.csv').dropna(subset=['text_nonstop'])

    documents = [doc.split() for doc in df['text_nonstop'].tolist()]
    print('calculating tfidf....')

    tfidf_res, id2token = tfidf(documents)
    with open('./result/id2token.pkl', 'wb') as f:
        pickle.dump(id2token, f)
    with open('./result/tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_res, f)

    if not Path('embeddings').is_dir():
        Path('embeddings').mkdir()

    for dim in [100, 200, 300]:
        print('%dd word2vec model training...' % dim)
        w2v_model = word2vec(documents, './embeddings/word2vec_%d.txt' % dim, dim, WINDOWS_SIZE, MIN_CNT)
        print(w2v_model.most_similar('环境'))
