'''
数据汇总和词频计算
'''

from pathlib import Path
import pandas as pd
import sys
import re
import string
from zhon.hanzi import punctuation
from utils import flatten, read_line, save_line
import collections

industry = read_line('industry.txt')
p = '(' + '|'.join([word for word in industry]) + ')'
stopwords = read_line('D:/Anaconda3/Lib/site-packages/pyhanlp/static/data/dictionary/stopwords.txt')
#    + list(string.punctuation) \
#    + list(punctuation)

def get_word(text):
    return [re.sub(', ', ' ', re.sub(r'/\w+', '', row[1:-1])) for row in text]

def get_pos(text):
    return [re.sub('/', '', ' '.join(re.findall(r'/\w+', row))) for row in text]

def word_count(text):
    non_stop_text = [[word for word in row.split() if word not in stopwords] for row in text]
    non_stop_text = [row for row in non_stop_text if len(row) > 3]
    valid_sentences = [sent for sent in non_stop_text if len(sent) / len(''.join(sent)) <= 0.7]
    word_cnt = collections.Counter(flatten(valid_sentences))
    invalid_sent_cnt = len(non_stop_text) - len(valid_sentences)
    valid_sentences = ' '.join(flatten(valid_sentences))
    return word_cnt, invalid_sent_cnt, valid_sentences

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
            text = get_word(read_line(file))
            try:
                word_cc, invalid_cnt, doc = word_count(text)
            except:
                print(file.name)
            counter = counter + word_cc
            region, industry, company = parse_file_name(file.name)
            all_content.append([str(i),  region, industry, company, invalid_cnt, ' '.join(text), doc, len(text)])

    df = pd.DataFrame(all_content)
    df.columns = ['year', 'location', 'industry', 'comp_name', 'invalid_sentences_cnt','text', 'text_nonstop', 'total_sents']
    df['invalid_ratio'] = df.invalid_sentences_cnt.values / df.total_sents.values
    df = df.drop(df[df.invalid_ratio > 0.2].index)
    df.to_csv('all_csr_text.csv', index=None)
    print('total documents', df.shape[0])

    words_cnt = {k:v for k,v in counter.items() if v >= 10}
    save_line(words_cnt, 'word_cnt.txt')