import itertools
import re
import collections
import pandas as pd

# 一些正则表达式


# 扁平化字符串列表
# ['1', '12', ['abc', 'df'], ['a']] ---> ['1','12','abc','df','a']
def flatten(x):
    tmp = [([i] if isinstance(i,str) else i) for i in x]
    return list(itertools.chain(*tmp))

# 多关键字匹配
def multiple_replace(text, adict, times=5):
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text, times)

class make_xlat:
    def __init__(self, *args, **kwds):
        self.adict = dict(*args, **kwds)
        self.rx = self.make_rx( )
    def make_rx(self):
        return re.compile('|'.join(map(re.escape, self.adict)))
    def one_xlat(self, match):
        return self.adict[match.group(0)]
    def __call__(self, text):
        return self.rx.sub(self.one_xlat, text)

def read_line(fname):
    return open(fname, encoding='utf8').read().split('\n')

def save_line(obj, fname):
    with open(fname, 'w', encoding='utf8') as f:
        if isinstance(obj, list):
            for item in obj:
                f.write(item+'\n')
        if isinstance(obj, collections.Counter) or isinstance(obj, dict):
            for key, val in sorted(obj.items(), key=lambda x: x[1], reverse=True):
                f.write(key + '\t' + str(val))
                f.write('\n')


def remove_stopwords(user_input, stop_words):
    stop_words = set(stop_words)
    for sw in stop_words.intersection(user_input):
        while sw in user_input:
            user_input.remove(sw)

    return user_input

def read_gram(fname, excel=False):
    gram = pd.read_csv(fname)
    ngram = int(fname[0])

    gram['gram_len'] = gram['gram'].apply(lambda x: len(x)-ngram+1)
    gram['word1'] = gram['gram'].apply(lambda x: x.split('-')[0])
    gram['word2'] = gram['gram'].apply(lambda x: x.split('-')[1])
    gram['word1_len'] = gram['word1'].apply(len)
    gram['word2_len'] = gram['word2'].apply(len)
    gram['pos1'] = gram['pos'].apply(lambda x: x.split('-')[0])
    gram['pos2'] = gram['pos'].apply(lambda x: x.split('-')[1])
    gram['pos1_abv'] = gram['pos1'].apply(lambda x: x[0] if x not in ['vn', 'an'] else x)
    gram['pos2_abv'] = gram['pos2'].apply(lambda x: x[0] if x not in ['vn', 'an'] else x)

    if ngram >= 3:
        gram['word3'] = gram['gram'].apply(lambda x: x.split('-')[2])
        gram['pos3'] = gram['pos'].apply(lambda x: x.split('-')[2])
        gram['word3_len'] = gram['word3'].apply(len)
        gram['pos3_abv'] = gram['pos3'].apply(lambda x: x[0] if x not in ['vn', 'an'] else x)
        
        if ngram == 4:
            gram['word4'] = gram['gram'].apply(lambda x: x.split('-')[3])
            gram['pos4'] = gram['pos'].apply(lambda x: x.split('-')[3])
            gram['word4_len'] = gram['word4'].apply(len)
            gram['pos4_abv'] = gram['pos4'].apply(lambda x: x[0] if x not in ['vn', 'an'] else x)
    
    if ngram >= 2:
        gram['pos_abv'] = gram['pos1_abv'].str.cat(gram['pos2_abv'].values, sep='-')
        
        if ngram >= 3:
            gram['pos_abv'] = gram['pos_abv'].str.cat(gram['pos3_abv'].values, sep='-')
            if ngram == 4:
                gram['pos_abv'] = gram['pos_abv'].str.cat(gram['pos4_abv'].values, sep='-')

    gram = gram[~gram['word%d'%ngram].str.match('[^\u4e00-\u9fa5]+')]
    gram = gram[~gram['word1'].str.match('[^\u4e00-\u9fa5]+')]
    gram = gram.fillna('')
    if excel:
        gram.to_excel('%dgram.xlsx'%ngram, index=None, header=True)
    else:
        gram.to_csv('%dgram_0515.csv'%ngram, index=None, header=True, encoding='utf8')

    return gram