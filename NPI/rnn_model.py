import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union
import sys
sys.path.append('/home/zdf/fage/nlp-beginer')
from NPI.utils import Vocab
from NPI.utils import loadWordEmbedding

class RNN(nn.Module):
    """
    实现了Reasoning about Entailment with Neural Attention中最简单的attention.
    """
    def __init__(self, embed_size, hidden_size, vocabList, device, dropout_rate=0.2):
        super(RNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = Vocab(vocabList)
        self.model_embeddings = loadWordEmbedding(self.vocab)
        self.device = device

        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.wy_projection = nn.Linear(hidden_size, hidden_size, bias=False)  #  called Wy in the paper
        self.wh_projection = nn.Linear(hidden_size, hidden_size, bias=False)  #  called Wh in the paper
        self.w_projeciton  = nn.Linear(hidden_size, 1, bias=False)  #  called w in the paper
        self.wp_projection = nn.Linear(hidden_size, hidden_size, bias=False)  #  called Wp in the paper
        self.wx_projection = nn.Linear(hidden_size, hidden_size, bias=False)  #  called Wx in the paper

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, premise: List[List[str]], hypothesis: List[List[str]]):
        premise_padded = self.vocab.to_input_tensor(premise, self.device, -1)  # torch.tensor (batch, seq1_len, embed_size)
        hypothesis_padded = self.vocab.to_input_tensor(hypothesis, self.device, -1) # torch.tensor (batch, seq2_len, embed_size)

        premise_emb = self.model_embeddings(premise_padded)
        hypothesis_emb = self.model_embeddings(hypothesis_padded)

        input_data = torch.cat((premise_emb, hypothesis_emb), dim=1)
        input_data = input_data.permute(1, 0, 2)  # torch.tensor (seq1_len+seq2_len, batch, embed_size)

        premise_max_len = self.get_max_seq(premise)

        premise_lengths = [len(it) for it in premise]
        mask = self.generate_sent_masks(premise_padded, premise_lengths)  # tensor, (batch, hidden_size)
        # 注意力机制
        r, h_n = self.attention(input_data, premise_max_len, mask)

        # called h* in the paper
        H = torch.tanh(self.wp_projection(r)+self.wx_projection(h_n))  # tensor, (batch, hidden_size)

        output = self.fc(H)

        return F.log_softmax(output, dim=1)


    def attention(self, X, max_len, mask):
        """
        注意力机制
        :param X: tensor, (seq1_len+seq2_len, batch, hidden_size)
        :param max_len: premise句子中最长的句子长度
        :param mask: tensor, (batch, seq1_len)
        :return:
        """
        # Y - tensor, (seq1_len+seq2_len, batch, hidden_size)
        Y, (h_n, c_n) = self.lstm(X)  # (h_0, c_0) 都初始化为0
        Y = Y[:max_len, :, :]  # tensor, (seq1_len, batch, hidden_size)
        h_n = torch.squeeze(h_n, 0)  # tensor, (batch, hidden_size)
        # c_n = torch.squeeze(c_n)

        eL = torch.ones(max_len, h_n.shape[0], h_n.shape[1])  # called eL in the paper
        eL = eL.to(self.device)

        # h_n*eL: tensor, (seq1_len, batch, hidden_size)  利用了python中的广播机制
        M = torch.tanh(self.wy_projection(Y) + self.wh_projection(h_n * eL))  # tensor, (seq1_len, batch, hidden_size)
        assert (M.shape == eL.shape)

        e_n = self.w_projeciton(M)   # tensor, (seq1_len, batch, 1)
        mask = mask.permute(1, 0).unsqueeze(-1)
        # 对于句子中的填充词，注意力为0
        e_n.masked_fill(mask.bool(), -float('inf'))

        alpha = F.softmax(e_n, dim=0)  # tensor, (seq1_len, batch, 1)
        assert (alpha.shape == (max_len, h_n.shape[0], 1))

        Y = Y.permute(1, 2, 0)  # tensor, (batch, hidden_size, seq1_len)
        alpha = alpha.permute(1, 0, 2)  # tensor, (batch, seq1_len, 1)

        r = torch.bmm(Y, alpha).squeeze(-1)  # tensor, (batch, hidden_size)

        return r, h_n


    def get_max_seq(self, sents: List[List[str]]):
        # 获取sents中最长的句子长度
        max_len = 0
        for it in sents:
            max_len = max(max_len, len(it))
        return max_len

    def generate_sent_masks(self, premise_padded, premise_lengths):
        """
        生成句子的mask, 句子中哪个地方使用了填充词, 就表示为1
        :param premise_padded: tensor, (batch, seq_len)
        :param premise_lengths: list[int], list中包含premise中每个句子长度
        :return:
            tensor, (batch, seq_len)
        """
        premise_mask = torch.zeros(premise_padded.shape[0], premise_padded.shape[1])
        for id, len in enumerate(premise_lengths):
            premise_mask[id, len:] = 1
        return premise_mask.to(self.device)
