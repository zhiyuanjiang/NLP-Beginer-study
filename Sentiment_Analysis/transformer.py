import torch
import torch.nn as nn
from utils import Vocab
from utils import loadWordEmbedding
import torch.nn.functional as F
import math

"""
code from : 
https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

代码中的multi-head attention的计算，不是将self-attention计算多次，然后进行拼接。
而是直接计算一个大的self-attention，这样就相当于计算多个小的self-attention，相当于multi-head attention，
速度更快。

最开始使用神经网络自动学习位置信息，但是网络基本上不拟合，后来使用正弦位置编码信息，网络才开始拟合。
但是效果一般般。

"""


def loadPosEmbedding(input_dim, d_model):
    """
    使用正弦位置编码
    input_dim: 最长的句子长度
    d_model: 相当于hidden dim
    return:
        embeddings - 返回一个位置编码的embedding层
    """
    weight = [[math.sin(pos / 10000 ** (i / d_model)) if i % 2 == 0 else
               math.cos(pos / 10000 ** (i / d_model)) for i in range(d_model) ] for pos in range(input_dim)]

    weights = torch.tensor(weight)
    embeddings = nn.Embedding.from_pretrained(weights, freeze=False)

    return embeddings

class Transformer(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 vocabList,
                 max_length=100):
        super().__init__()

        self.device = device
        self.vocab = Vocab(vocabList)
        self.embed_size = input_dim
        self.n_heads = n_heads

        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.tok_embedding = loadWordEmbedding(self.vocab)
        self.pos_embedding = loadPosEmbedding(max_length, hid_dim)


        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])


        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.drop = nn.Dropout(dropout)
        self.classify = nn.Linear(hid_dim, 5)

    def forward(self, src, src_mask=None):
        # src = list[batch size, src len]
        # src_mask = [batch size, src len]

        src = self.vocab.to_input_tensor(src, None, 60)
        batch_size = src.shape[0]
        seq_len = src.shape[1]

        zero = torch.zeros(1, )
        one = torch.ones(1, )
        src_mask = torch.where(src == 0., zero, one)

        src_mask = src_mask.repeat(self.n_heads*seq_len, 1).reshape(batch_size, self.n_heads, seq_len, seq_len)

        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        # src = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        src = torch.max(src, dim=1)[0]

        # src = [batch size, hid dim]

        src = self.classify(self.drop(src))

        return F.log_softmax(src, dim=1)


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        # mask = torch.unsqueeze(mask, -1)
        # mask = torch.unsqueeze(mask, -1)

        # print(energy.shape)
        # print(mask.shape)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x