"""提取名词的词向量"""
from data_combine import get_pos, get_word
from pathlib import Path
from utils import read_line, flatten
from collections import Counter
import string
import re

punc = list(string.punctuation)

def new_w2v(fname, nouns, dim, output_path):
    embeddings_index = {}
    with open(fname, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = str(values[0])
            if word in nouns and not word.isdigit() and word not in punc and len(word) > 1 and not re.search('[^\u4e00-\u9fa5]+', word):
                coefs = values[1:]
                embeddings_index[word] = coefs
    words_cnt = len(embeddings_index)
    with open(output_path, 'w', encoding='utf8') as f:
        f.write(str(words_cnt)+' '+str(dim)+'\n')
        for k,v in embeddings_index.items():
            row = k + ' ' + ' '.join(v)
            f.write(row+'\n')


if __name__ == '__main__':

    nouns = []
    gram = read_line('gram.txt')

    for i in range(2002, 2017):
        output_path = Path(__file__).parent / 'word_count' / str(i)
        if not output_path.exists():
            output_path.mkdir()
        print('reading files from folder', str(i))
        input_path = Path(__file__).parent / 'segmented' / str(i)
        input_files = input_path.glob('*.txt')
        for file in input_files:
            content = read_line(file)
            text = [row.split() for row in get_word(content)]
            pos = [row.split() for row in get_pos(content)]
            word_pos = flatten([list(zip(text[i], pos[i])) for i in range(len(text))])
            for item in word_pos:
                if item[1][0] in ['n', 'g'] or item[1][-1] == 'n':
                    nouns.append(item[0])
    cc = Counter(nouns)
    nouns = [k for k, v in cc.items() if v >= 10] + gram
    print('total nouns；%d'%len(nouns))

    print('generate only noun word2vec...')
    new_w2v('word2vec_100.txt', nouns, 100, 'word2vec_noun_100.txt')
    new_w2v('word2vec_200.txt', nouns, 200, 'word2vec_noun_200.txt')
    new_w2v('word2vec_300.txt', nouns, 300, 'word2vec_noun_300.txt')
