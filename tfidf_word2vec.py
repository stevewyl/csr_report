"""TFIDF计算和词向量训练"""

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim import corpora
from pyhanlp import HanLP
import pandas as pd
import re
from pathlib import Path
import pickle
from utils import read_line, remove_stopwords

WINDOWS_SIZE = 5
MIN_CNT = 10
KEYWORDS_CNT = 8
PHRASE_CNT = 10
REBUILD = True

def word2vec(text, fname, ndims, window_size, min_cnt=1):
    model = Word2Vec(text, min_count=min_cnt, window=window_size, size=ndims)
    model.wv.save_word2vec_format(fname, binary=False)
    return model

def to_dict(list_tuple, id2token):
    d = {id2token[item[0]]:item[1] for item in list_tuple}
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return d
        
def tfidf(text, rebuild=False):
    # 生成字典和向量语料 
    dictionary = Dictionary(text)
    id2token = {v:k for k,v in dictionary.token2id.items()}
    # 每一篇文档对应的稀疏向量
    # 向量的每一个元素代表了一个word在这篇文档中出现的次数
    if Path('corpus.mm').is_file() and rebuild == False:
        corpus = corpora.MmCorpus('corpus.mm')
    else:
        corpus = [dictionary.doc2bow(d) for d in text]
        corpora.MmCorpus.serialize('corpus.mm', corpus)
    tfidf_model = TfidfModel(corpus, id2word=dictionary)
    corpus_tfidf = tfidf_model[corpus]
    res = [to_dict(mat, id2token) for mat in corpus_tfidf]
    return res, id2token

def get_top_words(fname, tfidf_mat, topn=10):
    pass

if __name__ == "__main__":
    stopwords = read_line('D:/Anaconda3/Lib/site-packages/pyhanlp/static/data/dictionary/stopwords.txt')
    df = pd.read_csv('all_csr_text.csv')
    df = df.fillna('')
    df = df[df.text != '']

    # 去除停用词和标点
    documents = [doc.split() for doc in df['text_nonstop'].tolist()]
    '''
    print('filtering stopwords...')
    if Path('tfidf.pkl').is_file() and REBUILD == False:
        with open('tfidf.pkl', 'rb') as f:
            documents = pickle.load(f)
    else:
        #少量数据测试很快,大量数据时跑不通
        #documents = [remove_stopwords(doc, stopwords) for doc in documents]
        documents = [[word for word in doc if word not in stopwords] for doc in documents]
        with open('text.pkl', 'wb') as f:
            pickle.dump(documents, f)
    '''
    print('calculating tfidf....')
    if Path('tfidf.pkl').is_file() and REBUILD == False:
        with open('tfidf.pkl', 'rb') as f:
            tfidf_res = pickle.load(f)
    else:
        tfidf_res, id2token = tfidf(documents, REBUILD)
        with open('id2token.pkl', 'wb') as f:
            pickle.dump(id2token, f)
        with open('tfidf.pkl', 'wb') as f:
            pickle.dump(tfidf_res, f)

    for dim in [100,200,300]:
        print('%dd word2vec model training...'%dim)
        w2v_model = word2vec(documents, 'word2vec_%d.txt'%dim, dim, WINDOWS_SIZE, MIN_CNT)
        print(w2v_model.most_similar('环境'))

    #print('keyword extract...')
    #df['keywords'] = df['text'].apply(lambda x: '/'.join(list(HanLP.extractKeyword(x, KEYWORDS_CNT))))
    #print('phrase extract...')
    #df['phrase'] = df['text'].apply(lambda x: '/'.join(HanLP.extractPhrase(re.sub(' ','',x), PHRASE_CNT)))

    #df.to_csv('all_csr_info.csv', index=None, header=True)
