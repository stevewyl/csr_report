# -*- coding: utf-8 -*-
'''
csr报告文本清理脚本 Python 3.6
第一步：转换（全角，繁体）
第二步：清理（无效标点字符等，解析乱码）
第三步：整理合并（提取目录信息，目录前的文本全部删除）
第四步：筛选重要段落
'''
'''
代码改进方向：
1. 为丢失句号的段落结束增加分隔符(部分完成)
2. 优化识别连续句号结尾的两行行且没有空行的情况(部分完成)
3. 优化识别没有标点的行(部分完成)
4. 去除小标题行（是否有必要）(增加headline记号)
5. 去除不相关的内容，如报告的开头和结尾(部分完成)
'''

import sys
import re
from pathlib import Path
import glob
from tqdm import tqdm
from langconv import Converter
import more_itertools as mit
import itertools
from utils import flatten, multiple_replace
#from flashtext import KeywordProcessor

# 一些正则表达式
cn_pattern = r'[\u4e00-\u9fa5]+'
num_pattern = '[0-9]+'
en_pattern = '[a-zA-Z]+'
char_pattern = r'[a-zA-Z0-9\u4e00-\u9fa5]+'
valid_punc ='' 
# 初始化
#kk = KeywordProcessor()

'''
文件读取
'''
# 获取当前目录下所有的txt文件
def get_files(dir):
    try:
        data_path = Path(__file__).parent / dir
        filelist = data_path.glob('*.txt')
    except:
        data_path = './' + dir
        filelist = glob.glob(data_path + '/*.txt')
    return filelist

# 按行读入文件内容
def read_line(file):
    try:
        with open(file, 'r', encoding='utf8') as f:
            return [line.strip() for line in f.readlines()]
    except:
        print(file)


'''
转换
'''
# 全角转半角
# 句号不行
def strQ2B(line):
    rstring = ""  
    for uchar in line:
        inside_code = ord(uchar)  
        if inside_code == 12288:    #全角空格直接转换
            inside_code = 32  
        elif inside_code >= 65281 and inside_code <= 65374: #全角字符（除空格）根据关系转化
            inside_code -= 65248  
            rstring += chr(inside_code)
        else:
            rstring += chr(inside_code)
    return rstring

# 繁转简
def tra2sim(line):
    return Converter('zh-hans').convert(line)
# 转换
def transform(text):
    return [tra2sim(strQ2B(line)) for line in text]


'''
清理
'''
# 加载替换字列表
def load_keywords(fname):
    tmp = [item.split(',') for item in read_line(fname)]
    return {item[0]:item[1] for item in tmp}

# 去除名词短语+冒号开头的行
def check_colon(sent):
    if re.search(r'^[\u4e00-\u9fa5]+[^朋友]:', sent):
        return True
    else:
        return False

add_eos = lambda x: re.sub('。', '。<EOS>', x)

# 处理过长行
def cut_long_line(text):
    if len(text) > 1:
        splited = [add_eos(line).split('<EOS>') if len(re.findall('。', line)) > 2 and len(line) > 50 else line for line in text]
        return flatten(splited)
    else:
        return text

# 清理奇怪的字符
def clean_odd_symbol(text, keywords_dict):
    punc_pattern = r'[^a-zA-Z0-9\u4e00-\u9fa5 \.,!?\(\)。“、《》;:"\t]+'
    new = []
    for k,line in enumerate(text):
        line = re.sub(en_pattern, '', line) #去除英文
        line = multiple_replace(line, keywords_dict) #替换错别字
        line = re.sub('^ {4,}', '', line) #去除每一行开始多余的空格
        line = line+'(table)' if len(re.findall(r'\d+\t', line))>1 else line #添加表格标记
        line = re.sub(r'\.{3,}\d+|\t{2,}\d+|-{3,}\d+', '(catlog)', line) #替换为目录记号
        line = line+'(headline)' if re.search(r'^>', line) else line #替换为小标题记号
        line = re.sub(r'\r|\f|\v| {2,}', '', line) #去除空白字符
        line = re.sub(punc_pattern, '', line) #去除标点符号
        line = re.sub(r'\s(?=[^\(\)]*\))', '', line) #去除括号内的空格
        line = re.sub('。{2,}', '。', line)
        line = re.sub(r'(。|!|\?)', '\\1<EOS>', line) #增加分隔符标志
        line = re.sub(r'(\(headline\))', '\\1<EOS>', line) #增加分隔符标志
        line = re.sub(r'\(\)', '', line) #删除无内容的括号
        line = re.sub(r'\.{2,}', '', line)
        # 去除非标准空行
        if check_invalid_blank_line(k, line ,text):
            pass
        elif check_catlog(line):
            pass
        elif check_colon(line):
            pass
        elif re.search('\t', line):
            if re.search('^\t|\t$', line):
                new.append(line.replace('\t','').strip())
            else:
                pass
        elif re.search(r'\(([\u4e00-\u9fa5]+)\)$', line) or re.search(r'工时\)', line):
            pass
        elif re.search('报表|证券代码|年度社会责任报告|报告说明|目录|联系电话|联络电话|下载阅读', line):
            pass
        else:
            new.append(line.strip())
    return new

# 去除非标准空行
def check_invalid_blank_line(k, line, text):
    valid = not(check_end(text[k-1]) or bool(re.search('headline', text[k-1])))
    punc = check_punc(text[k-1])
    if k != len(text) - 1:
        length = abs(len(text[k-1]) - len(text[k+1])) < 5
        if valid and punc and length and line == '':
            return True
        else:
            return False
    else:
        return False

# 去除没有意义的开头内容
def get_start(text):
    idx = [k for k,v in enumerate(text) if re.search(r'致辞|尊敬的', v) and not check_catlog(v)]
    if idx != []:
        k = min(idx, key=lambda x:abs(x-len(text)*0.1))
        if k <= len(text)*0.15:
            return text[k+1:]
        else:
            return text
    else:
        return text

# 清理
def clean_transform(text, keywords_dict):
    text = cut_long_line(text)
    text = transform(text)
    text = get_start(text)
    text = clean_odd_symbol(text, keywords_dict)
    return text

'''
提取
'''
# 检查标点数量 
check_punc = lambda x: len(re.findall(r'、|。|,|!|\?|\(|\)|《|》|;', x)) > 0
# 检查没有标点的行的文本长度
check_no_punc = lambda x: len(re.findall(r'、|。|,|!|\?|\(|\)|《|》|;', x)) == 0 and len(x) > 10
# 检查是否包含中文
check_cn = lambda x: re.search(cn_pattern, x)
# 检查表格数字格式
check_num = lambda x: not re.search(r'\(table\)', x)
# 检查目录格式
check_catlog = lambda x: re.search(r'\(catlog\)', x)

# 提取标题的行号（未完成）
def get_headlines(text):
    pattern1 = '^[一|二|三|四|五]、' # 一、
    pattern2 = r'^\([一|二|三|四|五]\)' # （一）
    pattern3 = r'^\d+、' # 1、
    pattern4 = r'^\(\d+\)|\(headline\)' # (1)
    pattern5 = r'^[a-zA-Z0-9]{1,2}\.|^[a-zA-Z0-9]{1,2} |^0\d' # A. or 1.
    pattern6 = '^第[一|二|三|四|五]+部分'
    res = {'head1':[], 'head2':[], 'head3':[], 'head4':[]}
    for k,line in enumerate(text):
        if re.search(pattern1,line) and len(line) < 20: res['head1'].append(k)
        elif re.search(pattern2,line) and len(line) < 20: res['head2'].append(k)
        elif re.search(pattern3,line) and len(line) < 20: res['head3'].append(k)
        elif re.search(pattern4,line) and len(line) < 20: res['head4'].append(k)
        elif re.search(pattern5,line) and len(line) < 20: res['head4'].append(k)
        elif re.search(pattern6,line) and len(line) < 20: res['head1'].append(k)
    all_head = [ii for k,v in res.items() for ii in v]
    return res, all_head

# 获取目录信息（未完成）
def get_catlog(text):
    pattern1 = r'^目|^目录|^\s+目|CONTENTS'
    pattern2 = r'\.{1,}\d+$|\s{2,}\d+$'
    res = {'start':[], 'headline':[]}
    for k,line in enumerate(text):
        if re.search(pattern1,line): res['start'].append(k)
        elif re.search(pattern2,line): res['headline'].append(k)
    catlog = [re.findall(cn_pattern, text[i]) for i in res['headline']]
    return res, catlog

# 检查是否是合法的句子
def check_valid_sent(sent):
    return len(sent) > 10 and check_punc(sent) and sent[-1] not in ['。', '!', '?', '"']

# 检查末尾是否出现分隔符
def check_end(sent):
    return sent[-5:] == '<EOS>'

# 长度的特殊检查，去除EOS的长度
def check_length(sent):
    cnt_eos = re.findall('<EOS>', sent)
    if len(cnt_eos) > 0:
        return len(sent) - 4*len(cnt_eos)
    else:
        return len(sent)

# 多行段落
def check_long_sent(text, k):
    sent = text[k]
    if k == len(text)-2:
        before_after = bool(check_punc(text[k+1]))
    elif k == len(text)-1:
        before_after = False
    else:
        before_after = bool((text[k-1] == '' or check_punc(text[k-1]) or check_no_punc(text[k-1]))
                            and (check_punc(text[k+1]) or check_no_punc(text[k+1])))
    return len(sent) > 10 and sent[-5:] != '<EOS>' and before_after and check_cn(sent) and check_num(sent)

# 多行段落的最后一行
def check_last_line(text, k):
    sent = text[k]
    check_punc = re.findall(r'、|。|,|!|\?|\(|\)|《|》|;', sent)
    before = bool(check_long_sent(text, k-1))
    if k != len(text) - 1:
        if text[k+1] == '' and sent[-5:] != '<EOS>':
            length = bool(len(sent) <= len(text[k-1]))
            return len(check_punc) > 0 and before and check_cn(sent) and check_num(sent) and length
        else:
            return len(check_punc) > 0 and before and check_cn(sent) and sent[-5:] == '<EOS>' and check_num(sent)
    else:
        return False

# 单行段落
def check_single_line(text, k):
    sent = text[k]
    check_punc = re.findall(r'、|。|,|!|\?|\(|\)|《|》|;', sent)
    if k == len(text)-1:
        before_after = bool(text[k-1] == '')
    else:
        before_after = bool(text[k-1] == '' and text[k+1] == '')
    return check_length(sent) > 10 and len(check_punc) > 0 and check_cn(sent) and check_num(sent) and (sent[-5:] == '<EOS>' or before_after)

# 定义游标类
class cursor:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def reset(self):
        self.start = 0
        self.end = 0

    def begin(self, idx):
        self.start += self.start + idx
        self.end = self.start

    def add(self):
        self.end += 1

    def get_value(self):
        return(self.start, self.end)

# 进阶版（待开发）
def check_each_row(text):
    cc = cursor(0,0)
    res = []
    for k,row in enumerate(text):
        if check_valid_sent(row):
            if cc.get_value()[0] == 0:
                cc.begin(k)
            else:
                cc.add()
        if check_end(row):
            cc.add()
            res.append(cc.get_value())
            cc.reset()
    return res


'''
合并
'''
# 合并段落
'''
长度要求（大于10个字）
最后一个字符不是句号或感叹号或问号
遍历整个文本
'''
def combine_row(text):
    text.append('')
    sent_index = [k for k in range(len(text)) 
                  if check_long_sent(text, k) or check_single_line(text, k)
                  or check_last_line(text, k)]
    # 提取标题行
    _, all_head = get_headlines(text)
    # 去重
    sent_index = list(set(sent_index) - set(all_head))
    return list(find_ranges(sent_index))

# 对连续的index进行分组
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    iterable = sorted(iterable)
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]

# 多行段落的合并与拆分
def join_split(text, item):
    if type(item) == tuple:
        cut_text = ''.join(text[item[0]:item[-1]+1])
    else:
        cut_text = text[item]
    return re.split('<EOS>', cut_text)

# 输出结果
# 多行段落间每行没有间隔，单行段落需要加间隔
def output_paragraph(text, fname):
    index = combine_row(text)
    res = [join_split(text, item) for item in index]
    with open(fname, 'w', encoding = 'utf8') as f:
        for item in res:
            if len(item) > 1:
                for each in item:
                    f.write(re.sub('([\u4e00-\u9fa5]) ([\u4e00-\u9fa5])', '\\1\\2', each).strip())
                    f.write('\n')
                f.write('\n')
            elif len(item) == 1:
                f.write(re.sub('([\u4e00-\u9fa5]) ([\u4e00-\u9fa5])', '\\1\\2', item[0]).strip())
                f.write('\n')
                f.write('\n')


'''
失败结果汇总
'''
# 获取python对象的大小
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size               

# 转化失败结果汇总
def output_result_report(filelist1, filelist2, fname):
    origin, convert = [], []
    for file1, file2 in zip(filelist1, filelist2):
        if get_size(read_line(file1)) < 5000:
            origin.append(str(file1).split('\\')[-1])
        if get_size(read_line(file2)) / get_size(read_line(file1)) < 0.3:
            convert.append(str(file1).split('\\')[-1])
    with open(fname, 'w', encoding='utf8') as f:
        f.write('一些失败的转化结果汇总：')
        f.write('\n')
        f.write('\n')
        f.write('原始文件转化有误：')
        f.write('\n')
        for item in origin:
            f.write(item)
            f.write('\n')
        f.write('\n')
        f.write('丢失内容过多：')
        f.write('\n')
        for item in convert:
            f.write(item)
            f.write('\n')

if __name__ == '__main__':
    keywords_dict = load_keywords('./dict/keywords.txt')
    if not Path('cleaned').is_dir():
        Path('cleaned').mkdir()

    for i in range(2002, 2017):
        
        input_files = get_files(Path(__file__).parent / 'CSR_Texts' / str(i))
        output_path = Path(__file__).parent / 'cleaned' / str(i)
        
        if not output_path.exists():
            output_path.mkdir()
        print('reading files from folder', str(i))
        for file in tqdm(input_files):
            content = read_line(file)
            content = clean_transform(content, keywords_dict)
            fname = output_path.joinpath(file.name)
            output_paragraph(content, fname)
        
        # print('genearte preprocessing result report...')
        # output_result_report(input_files, get_files(output_path), '_'.join([str(i), 'result.txt']))