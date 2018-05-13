from pathlib import Path
from tqdm import tqdm
import sys
import re
from utils import flatten, multiple_replace

non_cn_pattern = r'[a-zA-Z0-9\.、_]+'
industry = open('industry.txt', encoding='utf8').read().split('\n')
adict = {word:' ' for word in industry}

def extract_name(name):
    name = re.sub(non_cn_pattern, '', name)
    name = re.sub('\(', '（', name)
    name = re.sub('\)', '）', name)
    #name = re.sub(r'\s+(?=[^《》]*》)', '', name) 
    name = re.sub('（\w+）$', '', name)
    name = multiple_replace(name, adict, 1)
    return name.strip()

# 获取所有CSR报告中的公司名
def get_files_name(year):
    folder = Path('CSR_Texts') / str(year)
    return [extract_name(file.name) for file in folder.glob('*.txt')]


if __name__ == '__main__':
    res = []
    for year in range(2002,2017):
        res.append(get_files_name(year))

    all_res = sorted(list(set(flatten(res))))
    
    with open('comp_names.txt', 'w', encoding='utf8') as f:
        for name in all_res:
            f.write(name)
            f.write('\n')