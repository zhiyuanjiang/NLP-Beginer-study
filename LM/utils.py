import csv
import copy
from gensim.models import KeyedVectors
from Vocab import Vocab
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import jieba


def loadRawData(filepath):
    """
    load data
    :param filepath:
    :return:
        data - List[List[str]]
    """
    with open(filepath, 'rb') as f:
        reader = f.readlines()
        cnt = 0
        data = []
        poetry = ""
        for sent in tqdm(reader):
            if cnt == 0:
                cnt += 1
                continue

            if sent == b'\r\n':
                continue
                # data.append(copy.copy(poetry))
                # poetry = ""
                # continue
            ss = str(sent, encoding='utf-8')
            ss = ss[:-2]
            # ss = re.subn("[{}]+".format("。，！"), "", ss)[0]
            # poetry += ss
            data.append(ss)
    return data

def processRawData(rawdata):
    data = []
    for sent in rawdata:
        s = "/".join(jieba.cut(sent, cut_all=False))
        data.append(s.split('/'))
    punction = "。，！；"
    data = [[word for word in sent if word not in punction] for sent in data]
    return data

def loadEmbeddings(vocab:Vocab, embed_size, filepath):
    """
    load embeddings
    :param vocab:
    :param embed_size: the size of embeddings
    :param filepath:
    :return:
    """
    w2v_model = KeyedVectors.load_word2vec_format(filepath, binary=False)
    # w2v_dict  = {word:vector for word,vector in zip(w2v_model.vocab, w2v_model.vectors)}

    vocab_size = vocab.len+1
    weights = torch.zeros(vocab_size, embed_size)

    word = []
    cnt = 0
    for i in range(vocab_size):
        if vocab.id2word[i] in w2v_model.vocab:
            weights[i, :] = torch.from_numpy(w2v_model[vocab.id2word[i]])
        else:
            word.append(vocab.id2word[i])
            cnt += 1

    embeddings = nn.Embedding.from_pretrained(weights)

    return embeddings

def batch_iter(data, batch_size, shuffle=False):

    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):

        indices = index_array[i * batch_size: (i + 1) * batch_size]
        new_data = [data[idx] for idx in indices]

        yield new_data

def data_split(data, test_size, random_state):
    # split data
    from sklearn.model_selection import train_test_split
    m = len(data)
    X = [i for i in range(m)]
    train_index,  test_index = train_test_split(X, test_size=test_size, random_state=random_state)

    train_x = [data[i] for i in train_index]
    test_x = [data[i] for i in test_index]

    return train_x, test_x

def loss_curve(loss_data):
    import matplotlib as mpl
    mpl.use('Agg')
    x = list(range(len(loss_data)))
    plt.plot(x, loss_data, marker='*')
    # plt.show()
    plt.savefig('./img/rnn.jpg')

def test_word2vec():
    filepath = './data/word2vec.6B.100d.txt'
    w2v_model = KeyedVectors.load_word2vec_format(filepath, binary=False)
    print(w2v_model['eu'])
    print(w2v_model.most_similar(['eu']))


if __name__ == '__main__':
    print('test')
    rawdata = loadRawData('./data/poetryFromTang.txt')
    data = processRawData(rawdata)
    for i in range(len(data)):
        print(data[i])
    print(len(data))
