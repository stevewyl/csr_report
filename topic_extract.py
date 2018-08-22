"""提取每篇文章的主题"""

import pandas as pd
from collections import defaultdict
import pickle
import sys
from utils import flatten


try:
    TOPIC_NUM = int(sys.argv[2])
except Exception:
    TOPIC_NUM = 10


def topic_sent(word_list, tfidf_dict, word_cluster):
    word_tfidf = {word: tfidf_dict[word] for word in word_list}
    topic_tfidf = defaultdict(list)
    for word in word_list:
        topic_tfidf[word_cluster[word]['CLUSTER_ID']].append(tfidf_dict[word])
    topic_tfidf = {k: sum(v) for k, v in topic_tfidf.items()}
    topic = sorted(topic_tfidf.items(), key=lambda x: x[1], reverse=True)[0]
    return {'topic': topic[0], 'avg_tfidf': round(sum(word_tfidf.values()) / len(word_tfidf), 4)}


def topic_word(word_list, tfidf_dict, word_cluster):
    word_list = list(set(word_list))
    topic_tfidf = defaultdict(list)
    for word in word_list:
        topic_tfidf[word_cluster[word]['CLUSTER_ID']].append(tfidf_dict[word])
    topic_num = {k: len(v) for k, v in topic_tfidf.items()}
    topic_tfidf = {k: sum(v) for k, v in topic_tfidf.items()}
    total_tfidf = sum(topic_tfidf.values())
    topic_tfidf = {k: [round(v / total_tfidf, 4), topic_num[k]] for k, v in topic_tfidf.items()}
    topic = sorted(topic_tfidf.items(), key=lambda x: x[1][0], reverse=True)
    return topic


print('load and prepare data')
word_cluster = pd.read_table(sys.argv[1]).set_index('WORD').to_dict(orient='index')
word_set = set(word_cluster.keys())
data = pd.read_csv('./result/all_csr_text.csv').dropna(subset=['text_nonstop'])
text = [line.strip().split('\t') for line in data['text_nonstop'].tolist()]

with open('./result/tfidf.pkl', 'rb') as f:
    tfidf_res = pickle.load(f)

tfidf_res = [{item[0]: item[1] for item in doc} for doc in tfidf_res]
words = [[list(set(sent.split(' ')) & word_set) for sent in doc] for doc in text]


# 计算方法 1
# 对于每个句子中的词，进行词主题的映射，计算每个主题的词的tfidf和，值最大的主题，作为这个句子的主题
# 这个句子的权重为该句子所有词的TFIDF的平均值
# 汇总每个主题的句子，按照每个主题的TFIDF值和进行排序，作为文章的主题
tfidf_sent = [[topic_sent(sent, tfidf_res[i], word_cluster) for sent in doc if len(sent) > 0] for i, doc in enumerate(words)]
final_res = []
for doc in tfidf_sent:
    topic = defaultdict(list)
    for sent in doc:
        topic[sent['topic']].append(sent['avg_tfidf'])
    topic_num = {k: len(v) for k, v in topic.items()}
    topic = {k: sum(v) for k, v in topic.items()}
    total_v = sum(topic.values())
    topic = {k: [round(v / total_v, 4), topic_num[k]] for k, v in topic.items()}
    topic = sorted(topic.items(), key=lambda x: x[1][0], reverse=True)
    topic_doc = {'topic_%d' % i: '' for i in range(1, TOPIC_NUM)}
    for i, item in enumerate(topic[:TOPIC_NUM]):
        topic_doc['topic_%d' % (i+1)] = str(item[0]) + '_' + str(item[1][0]) + '_' + str(item[1][1])
    final_res.append(topic_doc)
df_sent = pd.concat([data, pd.DataFrame(final_res)], axis=1).drop(['text', 'pos'], axis=1)
df_sent.to_csv('csr_topic_sent.csv', index=None, header=True)

# 计算方法 2
# 在词的级别来完成主题提取，计算每个主题对应的词的TFIDF和，每个主题的权重为该主题的TFIDF和的占比
f_words = [flatten(doc) for doc in words]
final_res = []
for i, doc in enumerate(f_words):
    topic = topic_word(doc, tfidf_res[i], word_cluster)
    topic_doc = {'topic_%d' % i: '' for i in range(1, TOPIC_NUM)}
    for i, item in enumerate(topic[:TOPIC_NUM]):
        topic_doc['topic_%d' % (i+1)] = str(item[0]) + '_' + str(item[1][0]) + '_' + str(item[1][1])
    final_res.append(topic_doc)
df_word = pd.concat([data, pd.DataFrame(final_res)], axis=1).drop(['text', 'pos'], axis=1)
df_word.to_csv('csr_topic_word.csv', index=None, header=True)
