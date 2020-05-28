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

"""
'I-MISC', 'B-MISC', 'I-LOC', 'B-LOC', 'I-ORG', 'B-ORG', 'I-PER', 'O'
'START' : 0
'I-MISC': 1
'B-MISC': 2
'I-LOC' : 3
'B-LOC' : 4
'I-ORG' : 5
'B-ORG' : 6
'I-PER' : 7
'O'     : 8
'END'   : 9
"""

def loadData(filepath):
    """
    load data
    :param filepath:
    :return:
        data - List[List[str]]
        labels - List[List[str]]
    """
    with open(filepath, 'r') as f:
        reader = f.readlines()
        cnt = 0
        data, labels = [], []
        sents, name = [], []
        print("load train data")
        for it in tqdm(reader):
            if cnt == 0 or cnt == 1:
                cnt += 1
                continue
            if it == '\n':
                data.append(copy.copy(sents))
                labels.append(copy.copy(name))
                sents.clear()
                name.clear()
                continue
            sp = it.split()
            sents.append(sp[0])
            name.append(sp[3])

    return data, labels

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

    for i in range(vocab_size):
        if vocab.id2word[i] in w2v_model.vocab:
            weights[i, :] = torch.from_numpy(w2v_model[vocab.id2word[i]])

    embeddings = nn.Embedding.from_pretrained(weights)

    return embeddings

def batch_iter(data, labels, batch_size, shuffle=False):

    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):

        indices = index_array[i * batch_size: (i + 1) * batch_size]
        new_data = [data[idx] for idx in indices]
        new_labels = [labels[idx] for idx in indices]

        yield new_data, new_labels

def data_split(data, labels, test_size, random_state):
    # split data
    from sklearn.model_selection import train_test_split
    m = len(labels)
    X = [i for i in range(m)]
    train_index,  test_index = train_test_split(X, test_size=test_size, random_state=random_state)

    train_x = [data[i] for i in train_index]
    train_y = [labels[i] for i in train_index]
    test_x = [data[i] for i in test_index]
    test_y = [labels[i] for i in test_index]

    return train_x, test_x, train_y, test_y

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
    # train_data_path = './data/conll2003/eng.train'
    # data, labels = loadData(train_data_path)
    # print('the size of data: {}, the size of labels: {}'.format(len(data), len(labels)))
    # print(data[:5])
    # print(labels[:5])
    # name = set()
    # for it in labels:
    #     name = name | set(it)
    # print(name)
    vocabList = ['zhi', 'yuan', 'jiang']
    vocab = Vocab(vocabList)
    loadEmbeddings(vocab, 100, './data/word2vec.6B.100d.txt')
