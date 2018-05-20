"""最终结果"""

import pandas as pd
from utils import read_line
from pprint import pprint
import re

stopwords_all = read_line('./stopwords_chunk/stopwords_all.txt')
stopwords_word1 = read_line('./stopwords_chunk/stopwords_word1.txt')
stopwords_last = read_line('./stopwords_chunk/stopwords_last_word.txt')
city = read_line('./stopwords_chunk/city.txt')

gram2 = pd.read_csv('2gram_0515.csv')
gram2 = gram2[gram2.score2 > 0.22]
gram2 = gram2[~gram2['word1'].isin(stopwords_word1)]
gram2 = gram2[~gram2['word1'].isin(stopwords_all)]
gram2 = gram2[~gram2['word2'].isin(stopwords_all)]
gram2 = gram2[~gram2['word2'].isin(stopwords_last)]
gram2 = gram2[~gram2['word2'].isin(city)]
gram2 = gram2[~gram2['pos1'].isin(['vn'])]
gram2 = gram2.drop(gram2[(gram2['word2'].isin(['有限公司', '集团', '公司', '股份有限公司'])) & (gram2['word1_len'] == 1)].index)
gram2['word2_1'] = gram2['word2'].apply(lambda x: x[0])
gram2['word1_2'] = gram2['word1'].apply(lambda x: x[-1])
gram2 = gram2.drop(gram2[(gram2['word2_1'] == gram2['word1']) | (gram2['word1_2'] == gram2['word2'])].index)
gram2['gram'] = gram2['gram'].apply(lambda x: re.sub('-','',x))
gram2 = gram2[['gram', 'score2']].sort_values('score2', ascending=False)['gram']
gram2.to_csv('2gram_cleaned.csv', index=None, header=True, encoding='utf8')

gram3 = pd.read_csv('3gram_0515.csv')
gram3 = gram3[gram3.score2 > 0.18]
gram3 = gram3[~gram3['word1'].isin(stopwords_word1)]
gram3 = gram3[~gram3['word1'].isin(stopwords_all)]
gram3 = gram3[~gram3['word2'].isin(stopwords_all)]
gram3 = gram3[~gram3['word3'].isin(stopwords_all)]
gram3 = gram3[~gram3['word3'].isin(stopwords_last)]
gram3 = gram3[~gram3['word3'].isin(city)]
gram3 = gram3[~gram3['pos1'].isin(['vn'])]
gram3 = gram3.drop(gram3[(gram3['word3'].isin(['有限公司', '集团', '公司', '股份有限公司'])) & (gram3['word1_len'] == 1)].index)
gram3['gram'] = gram3['gram'].apply(lambda x: re.sub('-','',x))
gram3 = gram3[['gram', 'score2']].sort_values('score2', ascending=False)['gram']
gram3.to_csv('3gram_cleaned.csv', index=None, header=True, encoding='utf8')

gram4 = pd.read_csv('4gram_0515.csv')
gram4 = gram4[gram4.score2 > 0.18]
gram4 = gram4[~gram4['word1'].isin(stopwords_word1)]
gram4 = gram4[~gram4['word1'].isin(stopwords_all)]
gram4 = gram4[~gram4['word2'].isin(stopwords_all)]
gram4 = gram4[~gram4['word3'].isin(stopwords_all)]
gram4 = gram4[~gram4['word4'].isin(stopwords_all)]
gram4 = gram4[~gram4['word4'].isin(stopwords_last)]
gram4 = gram4[~gram4['word4'].isin(city)]
gram4 = gram4[~gram4['pos1'].isin(['vn'])]
gram4 = gram4.drop(gram4[(gram4['word4'].isin(['有限公司', '集团', '公司', '股份有限公司'])) & (gram4['word1_len'] == 1)].index)
gram4['gram'] = gram4['gram'].apply(lambda x: re.sub('-','',x))
gram4 = gram4[['gram', 'score2']].sort_values('score2', ascending=False)['gram']
gram4.to_csv('4gram_cleaned.csv', index=None, header=True, encoding='utf8')

print('total number of chunks')
print('2gram:%d'%gram2.shape[0])
print('3gram:%d'%gram3.shape[0])
print('4gram:%d'%gram4.shape[0])
#pprint(gram2.sample(100)['gram'].tolist())
#pprint(gram3.sample(100)['gram'].tolist())
#pprint(gram4.sample(50)['gram'].tolist())