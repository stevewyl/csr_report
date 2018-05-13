# 使用HanLP的Python接口进行分词
'''
需要实现的功能：
1. 单文件分词
2. 多文件分词
3. 写成类方法
'''
'''
badcase 修复
1. 常见错误收集 ---> 整理了一部分错误词，已在清理阶段进行替换
2. 合并书名号内的文字《》
'''

from pyhanlp import HanLP, CustomDictionary, JClass
from pathlib import Path
import glob
import sys
import jieba
import re
from utils import save_line, read_line

'''
分词模块
对应结巴分词和HanLP提供的四种分词模型：
1.CRF（新词发现） 2.N最短路径（更精准） 3.感知机 4.标准分词器
'''
custom_dict = JClass('com.hankcs.hanlp.dictionary.CustomDictionary')
def cut_line(line_text):
    if sys.argv[1] == 'c':
        crf_segment = JClass('com.hankcs.hanlp.seg.CRF.CRFSegment')
        segment = crf_segment()
        segment.enableCustomDictionaryForcing(True)
        return [str(segment.seg(line)) for line in line_text if line != '']
    elif sys.argv[1] == 'n':
        n_short_segment = JClass('com.hankcs.hanlp.seg.NShort.NShortSegment')
        segment = n_short_segment()
        segment.enableCustomDictionaryForcing(True)
        return [str(segment.seg(line)) for line in line_text if line != '']
    elif sys.argv[1] == 'p':
        PerceptronLexicalAnalyzer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
        analyzer = PerceptronLexicalAnalyzer()
        return [str(analyzer.analyze(line)) for line in line_text if line != '']
    elif sys.argv[1] == 'j':
        jieba.load_userdict('C:/Users/stevewyl/OneDrive/中欧/csr/segment dictionary/csr.txt')
        return ['/ '.join(jieba.cut(line, cut_all=False)) for line in line_text if line != '']
    else:
        return [str(HanLP.segment(line)) for line in line_text if line != '']

'''
分词后清理
'''
def after_clean(line_seg):
    line_seg = re.sub('馀', '余', line_seg)
    return line_seg

if __name__ == '__main__':
    if not Path('segmented').is_dir():
        Path('cleaned').mkdir()

    for i in range(2002, 2017):
        print('reading files from folder', str(i))
        input_path = Path(__file__).parent / 'cleaned' / str(i)
        output_path = Path(__file__).parent / 'segmented' / str(i)
        if not output_path.exists():
            output_path.mkdir()
        input_files = input_path.glob('*.txt')
        for file in input_files:
            segmented_text = cut_line(read_line(file))
            save_line(segmented_text, output_path.joinpath(file.name))
