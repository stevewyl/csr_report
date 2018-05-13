"""
基于统计的方法寻找合适的NP
计算每个gram的分数分数有三部分组成：
	a. Gram的词频（0.2）
	b. Gram的点互信息（0.4）
    c. Gram的邻词信息熵（0.4）
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from data_combine import get_word, get_pos
from itertools import islice
from collections import Counter
from utils import read_line
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

FILTER_POS = ['w', 'c', 'u', 'm', 'q', 'p', 'f', 'r', 'b', 'i', 'k', 'l', 'z']
FILTER_VERB = ['vd', 'vf', 'vi', 'vg', 'vl', 'vshi', 'vyou', 'vx']
KEEP_POS = 'n'
NUM_GRAM = 50
CENTER_COUNT = 50
sys.argv[1] = 2

# N_Gram生成
def n_grams(a, n):
    z = (islice(a, i, None) for i in range(n))
    return zip(*z)

def loop_boolean(pos_filter, pos_list, mode='multi'):
    if mode == 'multi':
        bool_list = [[True if item == p else False for item in pos_list] for p in pos_filter]
    elif mode == 'single':
        bool_list = [[True if item == pos_filter else False for item in p] for p in pos_list]
    return any([any(item) for item in bool_list])


'''
过滤规则：
1. 不能包含标点或非字符符号
2. 任一词必须包含名词
3. 不能包含:助词、数词、连词、量词、介词、代词、成语、后缀等
'''
def filter_gram(line):
    length = len(line)
    for gram in n_grams(range(len(line)), sys.argv[1]):
        left = line[gram[0]-1][0] if gram[0] != 0 and line[gram[0]-1][1] != 'w' else ''
        right = line[gram[-1]+1][0] if gram[-1] != length-1 and line[gram[-1]+1][1] != 'w' else ''
        filter_pos_start = [line[i][1][0] for i in gram]
        pos_list = [line[i][1] for i in gram]
        keep_pos = [[line[i][1][0], line[i][1][-1]] for i in gram]
        if sys.argv[1] == 2:
            RULE_FILTER = loop_boolean(FILTER_POS, filter_pos_start)
            RULE_ONE_N = loop_boolean(KEEP_POS, keep_pos, 'single')
            RULE_VERB_ONE = (filter_pos_start[0] == 'v' or filter_pos_start[0] == 'n') and filter_pos_start[1] == 'v'
            RULE_VERB_TWO = loop_boolean(FILTER_VERB, pos_list)
            RULE_SAME = line[gram[0]][0] != line[gram[1]][0]
            RULE_NOUN = filter_pos_start[0] == 'n' and (filter_pos_start[1] in ['a', 'd', 't'])
            RULE_WORD1 = line[gram[0]][0] not in ['集团', '有限公司']
            if not RULE_FILTER and RULE_ONE_N and not RULE_VERB_ONE and not RULE_VERB_TWO and not RULE_NOUN and RULE_SAME and RULE_WORD1:
                bigram = line[gram[0]][0] + '-' + line[gram[1]][0]
                pos = line[gram[0]][1] + '-' + line[gram[1]][1]
                yield bigram, pos, line[gram[0]][0], line[gram[1]][0], left, right
        elif sys.argv[1] == 3:
            RULE_FILTER = loop_boolean(FILTER_POS, filter_pos_start)
            RULE_VERB_TWO = loop_boolean(FILTER_VERB, pos_list)
            RULE_TWO_NOUNS = len([p for p in filter_pos_start if p == 'n']) >= 2
            RULE_SAME = len(set([line[i][0] for i in gram])) == sys.argv[1]
            RULE_NOUN = filter_pos_start[-1] == 'n' or line[gram[-1]][1] == 'vn' or line[gram[-1]][1] == 'an'
            if not RULE_FILTER and RULE_TWO_NOUNS and not RULE_VERB_TWO and RULE_SAME and RULE_NOUN:
                trigram = line[gram[0]][0] + '-' + line[gram[1]][0] + '-' + line[gram[2]][0]
                pos = line[gram[0]][1] + '-' + line[gram[1]][1] + '-' + line[gram[2]][1]
                yield trigram, pos, line[gram[0]][0], line[gram[1]][0], line[gram[2]][0], left, right
            pass


if __name__ == '__main__':
    
    all_content = []
    print('get words and pos info...')
    for i in range(2002, 2017):
        print('read files from %d'%i)
        input_path = Path(__file__).parent / 'segmented' / str(i)
        input_files = input_path.glob('*.txt')
        for file in tqdm(input_files):
            text = [sent.split() for sent in get_word(read_line(file))]
            pos = [sent.split() for sent in get_pos(read_line(file))]
            res = [list(zip(text[j], pos[j])) for j in range(len(text))]
            for item in res:
                all_content.append(item)

    print('generate relative NP grams...')
    filter_gram = [x for row in tqdm(all_content) for x in filter_gram(row)]
    df = pd.DataFrame(filter_gram)
    if sys.argv[1] == 2:
        df.columns = ['gram', 'pos', 'word1', 'word2', 'left', 'right']
    if sys.argv[1] == 3:
        df.columns = ['gram', 'pos', 'word1', 'word2', 'word3', 'left', 'right']

    gram_cnt = Counter(df['gram'])
    word1_cnt = Counter(df['word1'])
    word2_cnt = Counter(df['word2'])
    if sys.argv[1] == 3:
        word3_cnt = Counter(df['word3'])
    word_cnt = word1_cnt + word2_cnt

    df['gram_count'] = df['gram'].map(gram_cnt)
    df['center_count'] = df['word2'].map(word2_cnt)
    df['word1_count'] = df['word1'].map(word_cnt)
    df['word2_count'] = df['word2'].map(word_cnt)
    if sys.argv[1] == 3:
        df['word3_count'] = df['word3'].map(word_cnt)
    df = df[(df['gram_count'] > NUM_GRAM) & (df['center_count'] > CENTER_COUNT)]
    total_cnt = df.shape[0]
    df.head(5)

    left_df = df[['gram', 'left', 'gram_count']]
    left_df = left_df[left_df.left != '']
    left_df['gram_left'] = left_df['left'].str.cat(left_df['gram'], sep='-')
    left_cnt = Counter(left_df['gram_left'])
    left_df = left_df.drop_duplicates(subset=['gram_left'])
    left_df['left_cnt'] = left_df['gram_left'].map(left_cnt)
    left_df['left_prob'] = left_df.left_cnt.values / left_df.gram_count.values
    left_df['left_entropy'] = - left_df.left_prob.values * np.log2(left_df.left_prob.values)
    left_df = left_df.groupby('gram')['left_entropy'].agg(sum).reset_index()

    right_df = df[['gram', 'right', 'gram_count']]
    right_df = right_df[right_df.right != '']
    right_df['gram_right'] = right_df['right'].str.cat(right_df['gram'], sep='-')
    right_cnt = Counter(right_df['gram_right'])
    right_df = right_df.drop_duplicates(subset=['gram_right'])
    right_df['right_cnt'] = right_df['gram_right'].map(right_cnt)
    right_df['right_prob'] = right_df.right_cnt.values / right_df.gram_count.values
    right_df['right_entropy'] = - right_df.right_prob.values * np.log2(right_df.right_prob.values)
    right_df = right_df.groupby('gram')['right_entropy'].agg(sum).reset_index()

    adjacent_df = pd.merge(df, left_df, on=['gram'])
    adjacent_df = pd.merge(adjacent_df, right_df, on=['gram'])

    df = adjacent_df.drop_duplicates(subset=['gram']).drop(['left','right', 'center_count'], axis=1)
    df['gram_count_log'] = np.log2(df['gram_count'].values)
    df['gram_prob'] = df['gram_count'].apply(lambda x: x / total_cnt)
    df['word1_prob'] = df['word1_count'].apply(lambda x: x / total_cnt)
    df['word2_prob'] = df['word2_count'].apply(lambda x: x / total_cnt)
    df['pmi'] = np.log2(df['gram_prob'].values / df['word1_prob'].values / df['word2_prob'].values)
    if sys.argv[1] == 3:
        df['word3_prob'] = df['word3_count'].apply(lambda x: x / total_cnt)
        df['pmi'] = np.log2(df['gram_prob'].values / df['word1_prob'].values / df['word2_prob'].values / df['word3_prob'].values)
    
    df = df.fillna(0)
    df['min_entropy'] = df.apply(lambda row: min(row['left_entropy'], row['right_entropy']), axis = 1)
    df['sum_entropy'] = df[['left_entropy', 'right_entropy']].sum(axis = 1)
    x = df[['gram_count_log', 'pmi', 'min_entropy', 'sum_entropy']]
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_new = pd.DataFrame(x_scaled)
    x_new.columns = ['wc_scaled', 'pmi_scaled', 'min_entropy_scaled', 'sum_entropy_scaled']
    df = pd.concat([df.reset_index(), x_new], axis=1)
    df['score1'] = df['wc_scaled'].values*0.4 + df['pmi_scaled'].values*0.4 + df['min_entropy_scaled'].values*0.2
    df['score2'] = np.cbrt(df['wc_scaled'].values * df['pmi_scaled'].values * df['min_entropy_scaled'].values)
    df = df.sort_values('score2', ascending=False)
    print('generate totally %s grams'%df.shape[0])
    df.to_csv('%dgram.csv'%sys.argv[1], index=None, header=True)
