import itertools
import re
import collections

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
