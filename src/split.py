import os
import sys
import jieba
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, os.pardir))
from src import settings

# step-1 章节切分  第.卷(.*)第.回
# with open(settings.HLM_TXT_PATH, 'r', encoding='utf-8') as f:
#     txt, chapter = '', 0
#     for line in f.readlines():
#         filename = settings.HLM_CHAPTER_PATH.format(chapter)
#         if '第' == line[0] and '卷' == line[2]:
#             w = open(filename, 'w', encoding='utf-8')
#             w.write(txt)
#             w.close()
#             chapter += 1
#             txt = ''
#         if line.startswith('红楼人物'):
#             w = open(filename, 'w', encoding='utf-8')
#             w.write(txt)
#             w.close()
#             break
#         txt += line

# # step-2 分词+词典
print(f'开始全文分词...')
vectorizer = CountVectorizer()          
with open(settings.HLM_TXT_PATH, 'r', encoding='utf-8') as f:
    words = jieba.lcut(f.read())        # 654136
    words = [word for word in words if 1 < len(word)]
    bag = vectorizer.fit_transform([' '.join(words)])   # 43054
print(f'分词结束，共获取{len(vectorizer.vocabulary_.keys())}个分词！')

# step-3 构建训练集
def dataset_generater(x, y, label, lx = [], ly = []):
    for i in range(x, y):
        filename = settings.HLM_CHAPTER_PATH.format(i)
        f = open(filename, 'r', encoding='utf-8')
        words = jieba.lcut(f.read()) 
        words = [word for word in words if 1 < len(word)]
        lx.append(' '.join(words))
        ly.append(label)
        f.close()    
    return lx, ly

train_x_data, train_y_data = dataset_generater(1, 41, 1)
train_x_data, train_y_data = dataset_generater(101, 121, 0, 
                            lx=train_x_data, ly=train_y_data)
train_x_data = vectorizer.transform(train_x_data).toarray()
# step-4 训练svm
svc = svm.SVC(kernel='linear', cache_size=4096)
svc.fit(train_x_data, train_y_data)

# step-5 构建测试集
test_x_data, test_y_data = dataset_generater(41, 81, 1)
test_x_data, test_y_data = dataset_generater(81, 101, 0,
                            lx=test_x_data, ly=test_y_data)
test_x_data = vectorizer.transform(test_x_data).toarray()

# step-5 预测
test_p_data = svc.predict(test_x_data)
print(sum(test_p_data[:60]))
print(sum(test_p_data[-20:]))
