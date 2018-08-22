import pandas as pd
import random
from utils import save_line
import sys

df = pd.read_table(sys.argv[1])
top_20_words = df.groupby('CLUSTER_ID').head(20).reset_index(drop=True)
res = {}
for i in range(200):
    words = top_20_words[top_20_words.CLUSTER_ID == i]['WORD'].tolist()
    rand_items = [random.sample(words, 5) for _ in range(5)]
    idx_list = [j for j in range(200) if j != i]
    for k in range(5):
        select_idx = random.choice(idx_list)
        str_idx = str(i) + '_' + str(select_idx)
        select_words = top_20_words[top_20_words.CLUSTER_ID == select_idx]['WORD'].tolist()
        r = random.choice(select_words)
        a = rand_items[k] + [r]
        random.shuffle(a)
        res[str_idx] = ' '.join(a) + '\t' + r

save_line(res, sys.argv[2])
