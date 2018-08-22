# CSR报告脚本使用说明

运行环境：Python 3.6

## 安装依赖包

```bash
pip install -r requirement.txt #pyhanlp首次使用会下载数据，请耐心等待
```

>若运行提示缺失某些包，请在命令行输入 pip install xxx 进行安装  
>请将数据存放于CSR_Texts目录下，子文件夹为年份  
>result文件夹存放一些中间结果

## 如何使用
1. 清理报告 python clean.py 数据存放于cleaned文件夹
2. 文本分词 python segment.py 数据存放于segmented文件夹
3. 数据整合 python data_combine.py 得到数据文件all_csr_text.csv和词频统计文件word_cnt.txt
4. 计算TF-IDF和词向量训练 python tfidf_word2vec.py --> 词向量存放在embeddings文件夹下，同时会生成TF-IDF的结果缓存文件
5. 获取只包含名词的词向量并归一化词向量 python get_noun_normed_w2v.py --> 处理过的词向量也存放在embeddings文件夹下
6. 词向量聚类 python kmeans.py --model ./embeddings/word2vec_noun_normed_100.txt --k 200 --output word_200_cluster.txt
7. 获取分组的词向量用于专家评审 python get_word_groups.py word_200_cluster.txt group_words.txt
8. 主题抽取 python topic_extract.py word_200_cluster.txt 50 (也可以不输入主题数，默认10) ---> 结果格式：114_0.3242_17 主题id_分数_个数

## 【注意】
>分词前，请将dict文件下的gram.txt和csr.txt 放至 D:\Anaconda3\Lib\site-packages\pyhanlp\static\data\dictionary\custom（视你的安装目录为准）  
>然后，删除该文件夹下的CustomDictionary.txt.bin，修改D:\Anaconda3\Lib\site-packages\pyhanlp\static\hanlp.properties中的开头为CustomDictionaryPath的行为下述行  
>CustomDictionaryPath=data/dictionary/custom/CustomDictionary.txt; 现代汉语补充词库.txt; 全国地名大全.txt ns; 人名词典.txt; 机构名词典.txt; 上海地名.txt ns; csr.txt n; gram.txt n;data/dictionary/person/nrf.txt nrf;

## 主题抽取的两种方法
### 计算方法 1
>对于每个句子中的词，进行词主题的映射，计算每个主题的词的tfidf和，值最大的主题，作为这个句子的主题  
>这个句子的权重为该句子所有词的TFIDF的平均值  
>汇总每个主题的句子，按照每个主题的TFIDF值和进行排序，作为文章的主题  

### 计算方法 2
>在词的级别来完成主题提取，计算每个主题对应的词的TFIDF和，每个主题的权重为该主题的TFIDF和的占比


## 其他文件
>抽取名词短语（分词后）python chunk.py  
>后处理 python chunk_result.py  
>处理好的文件位于 dict/gram.txt  

>LDA主题抽取：lda.py，效果比较差