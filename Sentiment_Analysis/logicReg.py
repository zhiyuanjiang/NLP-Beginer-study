import csv
from collections import defaultdict
import re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

"""
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
"""

"""
score:
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

def pre_process_data(data, vocabList, filePath):
    """
    如果之前没有处理过该数据
    则将数据处理成向量的形式, 并保存到指定路径, 返回向量化的数据
    如果处理过该数据，直接从文件中读取
    :param data:
    :return:
    """
    sentVec = []
    if os.path.exists(filePath):
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            print('获取句子的向量化数据（从文件中读取）:')
            for i, item in tqdm(enumerate(reader)):
                sentVec.append(list(map(int, item)))
    else:
        with open(filePath, 'w', newline='') as f:
            writer = csv.writer(f)
            print('获取句子的向量化数据（直接处理，并保存到文件中）：')
            for i, item in tqdm(enumerate(data)):
                item = bagOfWord2Vec(vocabList, item)
                sentVec.append(item)
                writer.writerow(item)
    return sentVec

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

def save_params(W, b, filePath):
    with open(filePath, 'w', newline='') as f:
        W = W.tolist()
        b = b.squeeze().tolist()
        writer = csv.writer(f)
        writer.writerows(W)
        writer.writerow(b)

def read_params(filePath):
    W, b = [], []
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        cnt = 0
        for it in reader:
            if cnt < 5:
                W.append([float(i) for i in it])
            else:
                b.append([float(i) for i in it])
            cnt += 1

    return np.array(W), np.array(b).T

def softmax(x):
    ex = np.exp(x)
    return x/np.sum(ex, axis=0)

def loss_func(output, y):
    m = output.shape[1]
    loss = output[y, range(m)]
    loss = np.log(loss)
    return -1./m*np.sum(loss)

def train(train_x, train_y, epoch, batch, lr):
    """
    训练w,b
    :param train_x: ndarray shape=(n, m)
    :param train_y: ndarray shape=(m,)
    :param epoch:
    :param batch:
    :param lr:
    :return:
    """
    n, m = train_x.shape
    if os.path.exists('./data/logicParams.csv'):
        W, b = read_params('./data/logicParams.csv')
    else:
        W = np.random.randn(5, n)/np.sqrt(n)
        b = np.zeros((5, 1))

    for i in range(epoch):
        for j in range(0, m, batch):
            s, e = j, j+batch
            if j+batch > m:
                e = m

            bm = e-s

            X = train_x[:, s:e]
            Y = train_y[s:e]
            Z = np.dot(W, X)+b
            A = softmax(Z)

            loss = loss_func(A, Y)
            if j == 0:
                print("the {} epoch, the {} batch the loss is : {}".format(i, j, loss))

            dz = A
            dz[Y.tolist(), range(e-s)] -= 1
            db = 1./bm*np.sum(dz, axis=1, keepdims=True)
            dw = 1./bm*np.dot(dz, X.T)

            assert dw.shape == (5, n)
            assert db.shape == (5, 1)

            W -= lr*dw
            b -= lr*db

    return W, b
            
def test(test_x, test_y, W, b):
    Z = np.dot(W, test_x)+b
    A = softmax(Z)
    predicts = np.argmax(A, axis=0)
    correct_count = np.sum(test_y == predicts)
    print("the correct rate is : ", 1.*correct_count/len(test_y))

def kaggleTest(test_data_vec, W, b, filePath):
    Z = np.dot(W, test_data_vec) + b
    A = softmax(Z)
    predict = np.argmax(A, axis=0)
    predict = predict.tolist()

    tid = [156061 + i for i in range(len(predict))]
    kaggle_data = list(zip(tid, predict))

    print('the test data count is : ', len(predict))
    # print(kaggle_data)
    # newline='', 就不会产生空行
    with open(filePath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PhraseId', 'Sentiment'])
        writer.writerows(kaggle_data)

if __name__ == "__main__":
    print("logistic regression")
    train_data, labels = loadDataSet("./data/train.tsv")

    vocabList = createVocabList(train_data)
    # train_data = pre_process_data(train_data, vocabList, './data/sen2vec.csv')

    # train_x, test_x, train_y, test_y = data_split(train_data, labels, 0.1, 42)

    flag = 0
    if flag:
        print('train')
        # W, b = train(np.array(train_x).T, np.array(train_y), 50, 64, 0.0001)
        # save_params(W, b, './data/logicParams.csv')
    else:
        W, b = read_params('./data/logicParams.csv')

    # test_x_vec = []
    # print('change test data to vector')
    # for i, it in tqdm(enumerate(test_x)):
    #     test_x_vec.append(bagOfWord2Vec(vocabList, it))
    #
    # test(np.array(test_x_vec).T, np.array(test_y), W, b)

    test_data, labels = loadDataSet("./data/test.tsv", 1)
    test_data_vec = []
    print('change test data to vector')
    for i, it in tqdm(enumerate(test_data)):
        test_data_vec.append(bagOfWord2Vec(vocabList, it))
    kaggleTest(np.array(test_data_vec).T, W, b, './data/kaggleDataReg.csv')