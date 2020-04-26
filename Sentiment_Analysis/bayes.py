import csv
from collections import defaultdict
import re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

"""
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
"""

"""
score: 0.59219
"""

def textParse(text):
    # 去除text中的标点，将所有大写字符变成小写字符，分割成单词
    # punctuation = '!,;:.?'
    # text = re.sub('[{}]+'.format(punctuation), ' ', text)
    return text.lower().split()

def loadDataSet(filePath, tick=0):
    """
    加载数据
    :param filePath: 文件路径
    :return:
        train_data - list[list[str]], 每个单词都是一个特征
        labels - list[int], 标签
    """
    with open(filePath, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        flag = 0
        train_data = []
        labels = []
        # index = set() # 统计最原始的长句，划分后的phrase暂时不统计
        for it in data:
            if flag == 0:
                flag += 1
                continue
            # if it[1] in index:
            #     continue
            # index.add(it[1])
            if tick == 0:
                labels.append(int(it[3]))
            train_data.append(textParse(it[2]))
    return train_data, labels

def data_split(data, labels, test_size, random_state):
    # 将data划分为train data and test data, 划分下标
    m = len(labels)
    X = [i for i in range(m)]
    train_index,  test_index = train_test_split(X, test_size=test_size, random_state=random_state)
    train_x = [data[i] for i in train_index]
    train_y = [labels[i] for i in train_index]
    test_x = [data[i] for i in test_index]
    test_y = [labels[i] for i in test_index]
    return train_x, test_x, train_y, test_y

def createVocabList(dataSet):
    # 使用train data创建一个词表，所有的特征
    vocabList = set() # 词表
    print('create vocab list')
    for id, data in tqdm(enumerate(dataSet)):
        vocabList = vocabList|set(data)
    return list(vocabList)

def bagOfWord2Vec(vocabList, inputSet):
    # 获取文档向量，使用词袋模型（单词重复统计）
    # exp: ['this', 'is', 'a'] --> [0, 1, 2]
    vec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] += 1
        else:
            print('the word {} is not in  my vocabulary'.format(word))
    return vec

def train(train_data_vec, labels):
    """
    # 获取P(ci), P(wi|ci)
    :param train_data_vec: ndarray shape=(m, n) - 训练数据的向量集
    :param labels: ndarray shape=(m,) - 标签
    :return:
        pw - ndarray shape=(5, n), P(wi|ci)
        pc - ndarray shape=(5,), P(ci)
    """
    (m, n) = train_data_vec.shape
    pw = np.ones((5, n)) # P(wi|ci) 分子初始化为1
    num_ci_word = np.full((5,), 2)  # 分母初始化为2
    num_c = np.zeros((5,))
    for i in range(m):
        pw[labels[i],:] += train_data_vec[i, :]
        num_ci_word[labels[i]] += np.sum(train_data_vec[i, :])
        num_c[labels[i]] += 1
    for i in range(5):
        pw[i, :] = pw[i, :]/num_ci_word[i]
    pc = num_c/m
    return np.log(pw), np.log(pc)

def test(test_data_vec, labels, pw, pc):

    output = np.dot(test_data_vec, pw.T)+pc
    predict = np.argmax(output, axis=1)

    correct_count = np.sum(labels == predict)

    print("correct rate is : ", 1.*correct_count/len(labels))

def kaggleTest(test_data_vec, pw, pc, filePath):

    output = np.dot(test_data_vec, pw.T)+pc
    predict = np.argmax(output, axis=1)

    tid = [156061+i for i in range(len(predict))]
    kaggle_data = list(zip(tid, predict.tolist()))

    print('the test data count is : ', len(predict))
    # print(kaggle_data)
    # newline='', 就不会产生空行
    with open(filePath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PhraseId', 'Sentiment'])
        writer.writerows(kaggle_data)


if __name__ == "__main__":
    print("bayes algrithm")
    train_data, labels = loadDataSet("./data/train.tsv")
    maxLen = 0
    for it in train_data:
        maxLen = max(maxLen, len(it))
    print('the max len is : ', maxLen)

    train_x, test_x, train_y, test_y = data_split(train_data, labels, 0.1, 42)
    vocabList = createVocabList(train_x)
    train_x_vec = []
    print('change train data to vector.')
    for i, it in tqdm(enumerate(train_x)):
        train_x_vec.append(bagOfWord2Vec(vocabList, it))
    pw, pc = train(np.array(train_x_vec), np.array(train_y))

    test_x_vec = []
    print('change test data to vector')
    for i, it in tqdm(enumerate(test_x)):
        test_x_vec.append(bagOfWord2Vec(vocabList, it))
    # test(np.array(test_x_vec), np.array(test_y), pw, pc)
    # kaggleTest(np.array(test_x_vec), pw, pc, './data/kaggleData.csv')

    test_data, labels = loadDataSet("./data/test.tsv", 1)
    test_data_vec = []
    print('change test data to vector')
    for i, it in tqdm(enumerate(test_data)):
        test_data_vec.append(bagOfWord2Vec(vocabList, it))
    kaggleTest(np.array(test_data_vec), pw, pc, './data/kaggleData.csv')