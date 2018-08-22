"""提取名词的词向量"""
from data_combine import get_pos, get_word
from pathlib import Path
from utils import read_line, flatten, save_line
from collections import Counter
import string
import re
import numpy as np

PUNC = list(string.punctuation)


def noun_filter(word):
    if not word.isdigit() and word not in PUNC and len(word) > 1 and not re.search('[^\u4e00-\u9fa5]+', word):
        return True
    else:
        return False


def noun_normed_embed(fname, output_path, nouns, dim):
    content = open(fname, encoding='utf8').read().split('\n')
    embeddings = content[1:]
    embeddings = [item.split() for item in embeddings if item != '']
    embeddings = {item[0]: np.asarray(item[1:], dtype='float32') for item in embeddings if item[0] != ' '}
    noun_words = list(set(embeddings.keys()) & set(nouns))
    print('filter non noun words')
    noun_embeddings = {word: embeddings[word] for word in noun_words if noun_filter(word)}
    print('norm embeddings')
    norm_embeddings = {k: v / np.linalg.norm(v) for k, v in noun_embeddings.items()}
    output = [k + ' ' + ' '.join([str(item) for item in list(v)]) for k, v in norm_embeddings.items()]
    output = [str(len(output)) + ' ' + str(dim)] + output
    save_line(output, output_path)


if __name__ == '__main__':
    nouns = []
    gram = read_line('./dict/gram.txt')

    for i in range(2002, 2017):
        print('reading files from folder', str(i))
        input_path = Path(__file__).parent / 'segmented' / str(i)
        input_files = input_path.glob('*.txt')
        for file in input_files:
            content = read_line(file)
            text = [row for row in get_word(content)]
            pos = [row for row in get_pos(content)]
            word_pos = flatten([list(zip(text[i], pos[i])) for i in range(len(text))])
            for item in word_pos:
                if item[1][0] in ['n', 'g'] or item[1][-1] == 'n':
                    nouns.append(item[0])
    cc = Counter(nouns)
    nouns = [k for k, v in cc.items() if v >= 10] + gram
    print('total nouns：%d' % len(nouns))

    print('generate only noun word2vec...')
    noun_normed_embed('./embeddings/word2vec_100.txt', './embeddings/word2vec_noun_normed_100.txt', nouns, 100)
    noun_normed_embed('./embeddings/word2vec_200.txt', './embeddings/word2vec_noun_normed_200.txt', nouns, 200)
    noun_normed_embed('./embeddings/word2vec_300.txt', './embeddings/word2vec_noun_normed_300.txt', nouns, 300)
