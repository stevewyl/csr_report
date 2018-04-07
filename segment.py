# 使用HanLP的Python接口进行分词
'''
需要实现的功能：
1. 单文件分词
2. 多文件分词
3. 写成类方法
'''

from pyhanlp import *
from pathlib import Path
import glob
from tqdm import tqdm
import sys

'''
文件读入
'''
def read_line(fname):
    return open(fname, encoding='utf8').read().split('\n')

'''
分词模块
'''
def cut_line(line_text):
    return [str(HanLP.segment(line)) for line in line_text if line != '']


'''
增加额外的字典
'''
def add_dict(fname):
    pass

'''
文件导出
'''
def save(fname, segment_text):
    with open(fname, 'w', encoding='utf8') as f:
        for row in segment_text:
            f.write(row)
            f.write('\n')

'''
被调用的模块
'''
def segment(self, parameter_list):
    pass

if __name__ == '__main__':
    input_path = Path(__file__).parent / 'cleaned' / sys.argv[1]
    output_path = Path(__file__).parent / 'segmented' / sys.argv[1]
    if not output_path.exists():
        output_path.mkdir()
    input_files = input_path.glob('*.txt')
    for file in tqdm(input_files):
        segmented_text = cut_line(read_line(file))
        save(output_path.joinpath(file.name), segmented_text)


