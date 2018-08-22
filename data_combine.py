'''
数据汇总和词频计算
'''

from pathlib import Path
import pandas as pd
import re
# import string
# from zhon.hanzi import punctuation
from utils import flatten, read_line, save_line
import collections

industry = read_line('./dict/industry.txt')
p = '(' + '|'.join([word for word in industry]) + ')'
STOPWORDS = read_line('./dict/stopwords.txt')
#    + list(string.punctuation) \
#    + list(punctuation)


def get_word(text):
    return [re.sub(', ', ' ', re.sub(r'/\w+', '', row[1:-1])).split() for row in text]


def get_pos(text):
    return [re.findall(r'/(\w+)', row) for row in text]


def word_count(text, pos_list):
    new_text, new_pos = collections.defaultdict(list), []
    for i, row in enumerate(text):
        if len(row) > 3:
            try:
                sent, pos = zip(*[(x, y) for x, y in zip(row, pos_list[i]) if x not in STOPWORDS])
            except Exception as e:
                print(e)
                sent, pos = [], []
            if len(sent) > 3:
                new_text['non_stop'].append(sent)
                if len(sent) / len(''.join(sent)) <= 0.7:
                    new_text['valid'].append(sent)
                    new_pos.append(pos)
    word_cnt = collections.Counter(flatten(new_text['valid']))
    invalid_sent_cnt = len(new_text['non_stop']) - len(new_text['valid'])
    valid_sentences = '\t'.join([' '.join(item) for item in new_text['valid']])
    valid_pos = '\t'.join([' '.join(item) for item in new_pos])
    return word_cnt, invalid_sent_cnt, valid_sentences, valid_pos


def parse_file_name(fname):
    fname = fname.split('.')[0]
    fname = re.sub('、', '', fname)
    fname = re.sub('(City|Province|Regi|nan)', '\\1<cut>', fname, 1)
    fname = re.sub(p, '\\1<cut>', fname, 1)
    fname = fname.split('<cut>')
    return fname[0], fname[1], fname[2]


if __name__ == '__main__':
    all_content = []
    counter = collections.Counter()
    for i in range(2002, 2017):
        print('reading files from folder', str(i))
        input_path = Path(__file__).parent / 'segmented' / str(i)
        input_files = input_path.glob('*.txt')
        for file in input_files:
            data = read_line(file)[:-1]
            text = get_word(data)
            pos = get_pos(data)
            try:
                word_cc, invalid_cnt, doc_text, doc_pos = word_count(text, pos)
            except Exception as e:
                print(e, file.name)
            counter = counter + word_cc
            region, industry, company = parse_file_name(file.name)
            text = '\t'.join([' '.join(item) for item in text])
            pos = '\t'.join([' '.join(item) for item in pos])
            all_content.append([str(i), region, industry, company, invalid_cnt,
                                text, doc_text, pos, doc_pos, len(text)])

    df = pd.DataFrame(all_content)
    df.columns = ['year', 'location', 'industry', 'comp_name', 'invalid_sentences_cnt',
                  'text', 'text_nonstop', 'pos', 'pos_nonstop', 'total_sents']
    df['invalid_ratio'] = df.invalid_sentences_cnt.values / df.total_sents.values
    print('total documents', df.shape[0])
    df = df.drop(df[df.invalid_ratio > 0.2].index)
    if not Path('result').is_dir():
        Path('result').mkdir()
    df.to_csv('./result/all_csr_text.csv', index=None)
    print('total documents', df.shape[0])

    words_cnt = {k: v for k, v in counter.items() if v >= 10}
    save_line(words_cnt, './result/word_cnt.txt')
