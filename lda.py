# -*- coding: utf-8 -*-
"""
lda文档主题
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
"""
import gensim
from gensim import models, corpora
from gensim.models import CoherenceModel
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import sys

def model_train(dictionary, corpus, texts, limit, model_name='basic', start=5, step=10):
    coherence_values = []
    model_list = []
    perplexity = []
    for num_topics in range(start, limit, step):
        print(num_topics)
        if model_name == 'mallet':
            mallet_path = 'D:/NLP/lda/mallet-2.0.8/bin/mallet'
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        else:
            model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
            perplexity.append(model.log_perplexity(corpus))
        model_list.append(model) 
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values, perplexity


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


def plot_cv(coherence_values):
    limit=100; start=5; step=10;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


if __name__ == "__main__":
    model_name = sys.argv[1]
    # 构建单词表
    df = pd.read_csv('all_csr_text.csv')
    text_o = [item for item in df['text_nonstop'].tolist() if isinstance(item, str)]
    pos_o = [item for item in df['pos_nonstop'].tolist() if isinstance(item, str)]
    text = [item.split(' ') for item in text_o]
    pos = [item.split(' ') for item in pos_o]
    keep = ['a', 'n', 'g', 'v', 'd', 'i', 'j']
    text = [[word for j, word in enumerate(doc) if pos[i][j][0] in keep] for i, doc in enumerate(text)]
    dic = corpora.Dictionary(text)
    corpus = [dic.doc2bow(words) for words in text]
    
    
    model_list, coherence_values, perplexity = model_train(dic, corpus, text, model_name=model_name,
                                                           start=5, limit=100, step=10)
    # plot_cv(coherence_values)


    # Select the model and print the topics
    best_idx = coherence_values.index(max(coherence_values))
    optimal_model = model_list[best_idx]
    model_topics = optimal_model.show_topics(formatted=False)
    pprint(optimal_model.print_topics(num_words=10))
    
    # Finding the dominant topic in each sentence
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=text_o)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.to_csv('./lda/dominant_topic_%s_lda.csv' % model_name, header=True, index=None)

    # Find the most representative document for each topic
    sent_topics_sorted = pd.DataFrame()
    sent_topics_out = df_topic_sents_keywords.groupby('Dominant_Topic')
    for i, grp in sent_topics_out:
        sent_topics_sorted = pd.concat([sent_topics_sorted, 
                                        grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                        axis=0)
    sent_topics_sorted.reset_index(drop=True, inplace=True)
    sent_topics_sorted.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    sent_topics_sorted.to_csv('./lda/representative_document_%s_lda.csv' % model_name, header=True, index=None)
    
    
    # Topic distribution across documents
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
    df_dominant_topics.to_csv('./lda/topic_distributiont_%s_lda.csv' % model_name, header=True, index=None)
