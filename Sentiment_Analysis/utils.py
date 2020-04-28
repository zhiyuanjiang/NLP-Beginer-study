import copy
import torch
from torch import nn
import numpy as np
import math

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
    glove_file = datapath('/home/zdf/fage/nlp-beginer/Sentiment_Analysis/data/glove.6B.100d.txt')
    # 指定转化为word2vec格式后文件的位置
    # tmp_file = get_tmpfile("f:/NLP-Beginner-study/Sentiment_Analysis/data/word2vec.6B.100d.txt")
    # tmp_file = get_tmpfile("/home/zdf/fage/nlp-beginer/Sentiment_Analysis/data/word2vec.6B.100d.txt")
    from gensim.scripts.glove2word2vec import glove2word2vec
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

