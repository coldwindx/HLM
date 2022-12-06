import numpy as np
import pandas as pd

import jieba
import jieba.posseg

import settings

pd.set_option('display.max_rows', None)
stopwords = pd.read_csv(
    settings.STOP_WORDS_PATH, 
    header=None, 
    names=["stop_words"]
)
dictionary = pd.read_csv(
    settings.HLM_DICT_PATH,
    header=None,
    names=['dictionary']
)
content = pd.read_csv(
    settings.HLM_TXT_PATH,
    header=None,
    names=['content']
)
# 删除空行
n = np.sum(pd.isnull(content))
# 提取每一回的标题
index = content.content.str.match('^第+.+回')
chapter_names = content.content[index].reset_index(drop=True)
chapter_names_split = chapter_names.str.split("[ 　]+").reset_index(drop=True)
# 创建数据集
data = pd.DataFrame(list(chapter_names_split), columns=['chapter', 'left_name', 'right_name'])
data['chapter_number'] = np.arange(1, 121)
data['chapter_name'] = data.left_name + ',' + data.right_name
data['start_id'] = index[index == True].index
data['end_id'] = data.start_id[1:len(data.start_id)].reset_index(drop=True) - 1
data["end_id"][[len(data["end_id"])-1]] = content.index[-1]
data['length_of_chapters'] = data.end_id - data.start_id
data['content'] = ''
for i in data.index:
    id = np.arange(data.start_id[i] + 1, data.end_id[i])
    data['content'][i] = ''.join(list(content.content[id])).replace("    ", '')
data['length_of_characters'] = data.content.apply(len)

# 全文分词
rows, cols = data.shape
data['words'] = ''
jieba.load_userdict(settings.HLM_DICT_PATH)
for i in range(rows):
    words = jieba.posseg.cut(data.content[i])
    words = pd.DataFrame(words, columns=['word', 'pos'])
    words = words.loc[(1 < words['word'].apply(len))
                         & (~words['word'].isin(stopwords))
                         & (words['pos'].isin(['n', 'nr', 'nr2', 'ns', 'nt', 'a', 'ad', 'an']))]
    data['words'][i] = words['word'].to_list()
data['length_of_words'] = data.words.apply(len)

# 存储数据集
data.to_json('./datasets/data.json')
