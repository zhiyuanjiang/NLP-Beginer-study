import csv
from collections import defaultdict
import re
import numpy as np
from tqdm import tqdm

from utils import loadDataSet
from utils import data_split
from utils import createVocabList

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