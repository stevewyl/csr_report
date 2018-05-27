# csr_report
CSR报告项目全流程

运行环境：Python 3.6

本项目主要分为以下几部分：

1. 清理文本内容

	这部分主要分为以下几个模块：
	1. 文件的批量读取
	2. 中文的转换（繁转简，全角转半角）
	3. 清理

		a. 去除无意义符号或词
		
		b. 替换转化错误的字（可在脚本根目录新建keywords.txt， 格式为每行 错误的字\t正确的字 如：曰	日）
		
		c. 去除非标准空行
		
		d. 去除报告的开头内容
		
		e. 去除表格内容
		
		f. 切分过长行
		
	4. 提取

		a. 多行段落
		
		b. 多行段落的最后一行
		
		c. 单行段落
		
		d. 目录信息（暂时不能使用）
		
		e. 标题信息（暂时不能使用）
		
	5. 合并

	6. 输出结果

		a. 清理后的文本（保存在输出目录）
		
		b. 转化失败的文件名（保存在当前目录下的result.txt）

	使用方法：python&nbsp;&nbsp;clean.py

2. 分词

	本项目采用HanLP的分词模块

	需要安装HanLP的python接口 具体安装方法参考官方repo：https://github.com/hankcs/pyhanlp
 
	使用方法：python&nbsp;&nbsp;segment.py&nbsp;&nbsp;s （这里的s参数表示标准分词，具体参见代码注释）

3. 短语提取

	针对分词器切词太散且实体词以完整的形式保留，能够尽可能表达语义。
	
	目前采取以下方式来提取合理的短语：
	1. n_gram的词性过滤提取+邻词生成
	2. 统计词频（gram，word1，word2，邻词），过滤gram词频和中心词词频
	3. 计算每个gram的点间互信息
	4. 计算每个gram的左右词的信息熵
	5. 将词频，互信息，邻词信息熵和放缩到[0,1]，加权求和，得到每个gram的分数

	使用方法：python&nbsp;&nbsp;NP_chunk.py&nbsp;&nbsp;n（n表示你需要生成多少个词组成的短语，目前只支持2和3）

	将生成的csv文件重新加入分词自定义字典中，进行分词

4. 过滤无效文本

	对错误句子数的比例超过20%的文档进行过滤

	错误句子定义为切词数目过多的句子，即切完词的词数占总字数的比例不能超过70%

5. TF-IDF
	
	计算每个词在该文档中的TF-IDF值

	使用方法：python&nbsp;&nbsp;tfidf_word2vec.py

6. Word2Vec

	训练100维、200维和300维的词向量，词窗大小为5，词频过滤为10

	使用方法：python&nbsp;&nbsp;tfidf_word2vec.py

7. Kmeans词向量聚类

	使用kmeans对词向量进行聚类，输出结果为某一类的词的集合文档

	使用方法：python&nbsp;&nbsp;kmeans.py -model xxxx -k xx -format 0 -output xxx

	model参数为词向量文件，k参数为聚类个数，output参数为输出文件名

# TODO
1. 聚类结果的优化

	a. 只针对名词进行聚类（完成）

	b. 尝试除Kmeans外的其他方法（接下来目标）

3. 短语提取模块的结果优化（完成）
