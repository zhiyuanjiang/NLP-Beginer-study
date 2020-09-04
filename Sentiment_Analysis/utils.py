import copy
import torch
from torch import nn
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

def textParse(text):
    # 去除text中的标点，将所有大写字符变成小写字符，分割成单词
    # punctuation = '!,;:.?'
    # text = re.sub('[{}]+'.format(punctuation), ' ', text)
    return text.lower().split()

def loadDataSet(filePath, tick=0):
    """
    加载数据
    :param filePath: 文件路径
    :param tick: tick=0 读取train.tsv, tick=1 读取test.tsv
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
    # return list(vocabList)
    return vocabList

class Vocab(object):

    def __init__(self, vocabList):
        self.word2id = {word: i + 1 for i, word in enumerate(vocabList)}
        # 定义了一个unknown的词，也就是说没有出现在训练集里的词，我们都叫做unknown，词向量就定义为0。
        self.word2id['<unk>'] = 0
        self.id2word = {i + 1: word for i, word in enumerate(vocabList)}
        self.id2word[0] = '<unk>'
        self.len = len(vocabList)

    def getWordId(self, word):
        if word not in self.word2id:
            return self.word2id['<unk>']
        else:
            return self.word2id[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self.getWordId(w) for w in s] for s in sents]
        else:
            return [self.getWordId(w) for w in sents]
    
    def to_input_tensor(self, sents, device, max_sent_len):
        """
        将sents转换成tensor
        :param sents: list[list[str]]
        :param device:
        :return:
            - shape=(batch, max_sent_len)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self.word2id['<unk>'], max_sent_len)
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return sents_var


def pad_sents(sents, pad_token, max_sent_len):
    """
    在batch中根据最长的句子长度来对所有的句子进行填充，使得batch中的所有句子长度一样。
    单词应该填充在所有句子末尾。
    :param sents: list[list[str]] , 句子构成的list，其中每个句子是由多个单词构成的list
    :param pad_token: 填充的word
    :param max_sent_len: 最长的句子长度
    :return:
        - list[list[str]] 填充后的sents
    """
    sents_padded = copy.deepcopy(sents)

    # 当max_sent_len = -1时，此时是RNN模型调用该函数，不需要提前确定最长的句子长度
    if max_sent_len == -1:
        for s in sents_padded:
            max_sent_len = max(len(s), max_sent_len)

    for s in sents_padded:
        s.extend([pad_token]*(max_sent_len-len(s)))

    return sents_padded


def loadWordEmbedding(vocab: Vocab):
    """
    加载预训练好的词向量
    :return:
        embeddings - 训练好的词向量
    """

    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors
    # 已有的glove词向量
    # glove_file = datapath('f:/NLP-Beginner-study/Sentiment_Analysis/data/glove.6B.100d.txt')
    # glove_file = datapath('/home/zdf/fage/nlp-beginer/Sentiment_Analysis/data/glove.6B.100d.txt')
    # 指定转化为word2vec格式后文件的位置
    # tmp_file = get_tmpfile("f:/NLP-Beginner-study/Sentiment_Analysis/data/word2vec.6B.100d.txt")
    # tmp_file = get_tmpfile("/home/zdf/fage/nlp-beginer/Sentiment_Analysis/data/word2vec.6B.100d.txt")
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # glove2word2vec(glove_file, tmp_file)

    tmp_file = "/home/zdf/fage/nlp-beginer/Sentiment_Analysis/data/word2vec.6B.100d.txt"
    word2vec_model = KeyedVectors.load_word2vec_format(
        tmp_file, binary=False, encoding='utf-8')
    word2vec_map = {word: vector for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors)}
    vocab_size = vocab.len + 1
    embed_size = 100

    weights = torch.zeros(vocab_size, embed_size)
    for i in range(vocab_size):
        if vocab.id2word[i] in word2vec_map:
            weights[i, :] = torch.from_numpy(word2vec_map[vocab.id2word[i]])

    # freeze=False, when update, fine-tune weights
    embeddings = nn.Embedding.from_pretrained(weights, freeze=False)

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

def batch_iter_test(data, batch_size):

    batch_num = math.ceil(len(data)/batch_size)
    index_array = list(range(len(data)))

    for i in range(batch_num):

        indices = index_array[i*batch_size: (i+1)*batch_size]
        new_data = [data[idx] for idx in indices]

        yield new_data


def createPreTrainVocab():
    """
    获取词向量的中的所有词，构成一个词表
    :return:
    """
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors

    # 已有的glove词向量
    glove_file = datapath('f:/NLP-Beginner-study/Sentiment_Analysis/data/glove.6B.100d.txt')
    # glove_file = datapath('/home/yuan/nlp-beginer/Sentiment_Analysis/data/glove.6B.100d.txt')
    # 指定转化为word2vec格式后文件的位置
    tmp_file = get_tmpfile("f:/NLP-Beginner-study/Sentiment_Analysis/data/word2vec.6B.100d.txt")
    # tmp_file = get_tmpfile("/home/yuan/nlp-beginer/Sentiment_Analysis/data/word2vec.6B.100d.txt")
    from gensim.scripts.glove2word2vec import glove2word2vec

    glove2word2vec(glove_file, tmp_file)

    word2vec_model = KeyedVectors.load_word2vec_format(
        tmp_file, binary=False, encoding='utf-8')
    # word2vec_map = {word: vector for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors)}

    vocab = word2vec_model.vocab.keys()

    # return list(vocab)
    return set(vocab)

def loss_curve(loss_data):
    x = list(range(len(loss_data)))
    plt.plot(x, loss_data, marker='*')
    # plt.show()
    plt.savefig('./img/rnn.jpg')
