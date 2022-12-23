
import collections
import numpy as np
import pandas as pd
import jieba
import jieba.posseg

import settings

pd.set_option('display.max_rows', None)
stopwords = pd.read_csv(
    settings.STOP_WORDS_PATH, 
    header=None, 
    names=["stop_words"],
    delimiter="\\n"
)
content = pd.read_csv(
    settings.HLM_TXT_PATH,
    header=None,
    names=['content'],
    delimiter="\\n"
)
# 删除空行
n = np.sum(pd.isnull(content))

# 提取每一回的标题
index = content.content.str.match('^第+.+回 ')
chapter_names = content.content[index].reset_index(drop=True)
chapter_names_split = chapter_names.str.split("[ 　]+").reset_index(drop=True)
# 创建数据集
data = pd.DataFrame(list(chapter_names_split), columns=['chapter', 'left_name', 'right_name'])
data['chapter_number'] = np.arange(1, 101)
data['chapter_name'] = data.left_name + ',' + data.right_name
data['start_id'] = index[index == True].index
data['end_id'] = data.start_id[1:len(data.start_id)].reset_index(drop=True) - 1
data["end_id"][[len(data["end_id"])-1]] = content.index[-1]
data['length_of_chapters'] = data.end_id - data.start_id
data['content'] = data['words'] = ''
for i in data.index:
    id = np.arange(data.start_id[i] + 1, data.end_id[i])
    data['content'][i] = ''.join(list(content.content[id])).replace("    ", '')
data['length_of_characters'] = data.content.apply(len)

# 全文分词

unite = ["行者","大圣","老孙","悟空","师兄","猴王","齐天大圣","孙行者","孙大圣", 
            "孙悟空","大师兄", "猴子", "猢狲", "弼马温"]
dictionary = {'relevant': [], 'irrelevant': []}
rows, cols = data.shape
# jieba.load_userdict(settings.HLM_DICT_PATH)
for i in range(rows):
    sequences = data['content'][i].split("。")
    data['words'][i] = list()
    for seq in sequences:
        cuts = jieba.posseg.lcut(seq)
        cuts = pd.DataFrame(cuts, columns=['word', 'pos'])
        cuts = cuts.loc[(1 < cuts['word'].apply(len))
                            & (~cuts['word'].isin(stopwords['stop_words']))
                            & (cuts['pos'].isin(['n', 'nr', 'a', 'xc']))]
        if 0 == len(cuts):
            continue
        # 找到属于孙悟空的句子
        sign = any([seq.find(key) != -1 for key in unite])
        data['words'][i].append((sign, cuts['word'].to_list()))
        if sign:
            dictionary['relevant'] += cuts['word'].to_list()
        else:
            dictionary['irrelevant'] += cuts['word'].to_list()

# CHI提取词典
relevant = pd.DataFrame(dictionary['relevant'], columns=['word'])
relevant = relevant['word'].value_counts().to_frame()
relevant = relevant.reset_index().rename(columns={'word': 'freq', 'index': 'word'})

irrelevant = pd.DataFrame(dictionary['irrelevant'], columns=['word'])
irrelevant = irrelevant['word'].value_counts().to_frame()
irrelevant = irrelevant.reset_index().rename(columns={'word': 'freq', 'index': 'word'})

total_relevant = int(relevant.sum(axis = 0, skipna = True)['freq'])
total_irrelevant = int(irrelevant.sum(axis = 0, skipna = True)['freq'])

chi = collections.defaultdict()
for word in relevant['word']:
    try:
        A = int(relevant[relevant['word'] == word]['freq'])
        B = int(irrelevant[irrelevant['word'] == word]['freq'])
        C = total_irrelevant - A
        D = total_irrelevant - B
        chi[word] = pow(A * D - B * C, 2) / ((A + B) * (C + D))
    except:
        chi[word] = 0

dictionary = sorted(chi.items(), key=lambda x:x[1], reverse=True)
print(f'dictionary len is {len(dictionary)}, and dictionary is:')
print(*dictionary, sep='\n')
dictionary = np.array(dictionary)[:2048, 0].tolist()

# 按词典分词
rows, cols = data.shape

for i in range(rows):
    t_words = data['words'][i]
    data['words'][i] = []
    for sign, words in t_words:
        if not sign:
            continue
        data['words'][i] += [w for w in words if w in dictionary]
# 存储数据集
data.to_json('./datasets/data.json')
