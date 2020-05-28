import copy
import torch

class Vocab(object):

    def __init__(self, vocabList):
        self.word2id = {word: i + 1 for i, word in enumerate(vocabList)}
        # 定义了一个unknown的词，也就是说没有出现在训练集里的词，我们都叫做unknown，词向量就定义为0。
        self.word2id['<unk>'] = 0
        self.id2word = {i + 1: word for i, word in enumerate(vocabList)}
        self.id2word[0] = '<unk>'
        self.pad_token = '<unk>'
        self.pad_token_id = 0
        self.len = len(vocabList)

    def getWordId(self, word):
        if word not in self.word2id:
            return self.pad_token
        else:
            return self.word2id[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self.getWordId(w) for w in s] for s in sents]
        else:
            return [self.getWordId(w) for w in sents]

    def to_input_tensor(self, sents):
        """
        将sents转换成tensor
        :param sents: list[list[str]]
        :param device:
        :return:
            - shape=(batch, max_sent_len)
        """
        word_ids = self.words2indices(sents)
        sents_t = self.pad_sents(word_ids, self.pad_token_id)
        sents_var = torch.tensor(sents_t, dtype=torch.long)
        return sents_var

    def pad_sents(self, sents, pad_token):
        """
        在batch中根据最长的句子长度来对所有的句子进行填充，使得batch中的所有句子长度一样。
        单词应该填充在所有句子末尾。
        :param sents: list[list[str]] , 句子构成的list，其中每个句子是由多个单词构成的list
        :param pad_token: 填充的word
        :return:
            - list[list[str]] 填充后的sents
        """
        sents_padded = copy.deepcopy(sents)
        max_sent_len = 0

        for s in sents_padded:
            max_sent_len = max(len(s), max_sent_len)

        for s in sents_padded:
            s.extend([pad_token] * (max_sent_len - len(s)))

        return sents_padded



def main():
    vocabList = ['zhi', 'yuan', 'jiang']
    vocab = Vocab(vocabList)
    device = torch.device('cpu')
    print(vocab.to_input_tensor([['zhi', 'yuan', 'yuan'], ['jiang']], device))



if __name__ == '__main__':
    main()
