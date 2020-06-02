import copy
import torch

class Vocab(object):

    def __init__(self, vocabList):
        self.word2id = {word: i for i, word in enumerate(vocabList)}

        self.word2id['<START>'] = len(self.word2id)
        self.word2id['<END>'] = len(self.word2id)
        self.word2id['<PAD>'] = len(self.word2id)
        self.word2id['<UNK>'] = len(self.word2id)

        self.id2word = {value: key for key, value in self.word2id.items()}

        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        self.pad_token_id = self.word2id['<PAD>']
        self.start_token_id = self.word2id['<START>']
        self.end_token_id = self.word2id['<END>']
        self.unk_token_id = self.word2id['<UNK>']
        self.len = len(self.word2id)

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
