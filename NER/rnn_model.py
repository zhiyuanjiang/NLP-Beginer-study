import torch
import torch.nn as nn
import torch.nn.functional as F
from Vocab import Vocab

class RNN(nn.Module):
    """
    from https://github.com/createmomo/CRF-Layer-on-the-Top-of-BiLSTM
    classify_size - the number of classification
    """

    def __init__(self, embed_size, hidden_size, vocabList, device):
        super(RNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = Vocab(vocabList)
        self.device = device

        self.lstm = nn.LSTM(embed_size, hidden_size)
        # the params to be add model, transition matrix
        self.transition = nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, inputs, labels):
        """
        first test batch = 1
        :param inputs: tensor, (batch, seq_len, embed_size)
        :param labels: tensor, (batch, seq_len)
        :return:
        """
        inputs = inputs.permute(1, 0, 2)  # (seq_len, batch, embed_size)
        # calculate prob
        transition = F.softmax(self.transition, dim=1)

        m = inputs.shape[0]
        n = inputs.shape[1]
        x, (h_n, c_n) = self.lstm(inputs)  # (seq_len, batch, hidden_size)
        x = F.softmax(x, dim=2)
        x = x.permute(1, 0, 2) # (batch, seq_len, hidden_size)

        d = torch.tensor(range(n)).reshape(n, 1)
        # the score of real path
        Pr = torch.sum(x[d, range(m), labels], dim=1) + \
             torch.sum(transition[labels[:, :-1], labels[:, 1:]], dim=1) # (batch, )

        # the score of all path
        Ps = self.CRF(x)  # (batch ,)
        loss = Ps-Pr
        loss = loss.sum()

        return loss

    def CRF(self, H):
        """
        calculate the score of all path
        :param H: (batch, seq_len, hidden_size)
        :return:
            scores - tensor, (batch, )
        """
        m = H.shape[1]
        transition = F.softmax(self.transition, dim=1)
        # first, calculate the score of 'start' to 'x1'
        pre_x = H[:, 0, :] + transition[0, :]   # (batch, hidden_size)
        pre_x = pre_x.unsqueeze(1)  # (batch, 1, hidden_size)
        pre_x = pre_x.permute(0, 2, 1)  # (batch, hidden_size, 1)

        # tmp = torch.full((H.shape[0], H.shape[2], H.shape[2]), 25).to(self.device)
        for i in range(1, m):
            Z = pre_x + transition + H[:, i, :].unsqueeze(1)   # (batch, hidden_size, hidden_size)
            # to avoid the score overflow
            pre_x = self.log_sum_exp(Z)

        # pre_x = pre_x.squeeze(-1)  # (batch, hidden_size)
        # finally, calculate the score of 'xn' to 'end'
        scores = pre_x + transition[-1, :].reshape(-1, 1)  # (batch, hidden_size, 1)
        scores = self.log_sum_exp(scores)
        scores = scores.squeeze()

        return scores

    def log_sum_exp(self, Z):
        """
        :param Z: (batch, hidden_size, hidden_size)
        :return:
        """
        max_val = torch.max(Z, dim=1, keepdim=True)[0]  # (batch, 1, hidden_size)
        pre_x = torch.sum(torch.exp(Z-max_val), dim=1, keepdim=True)  # (batch, 1, hidden_size)
        pre_x = torch.log(pre_x) + max_val
        pre_x = pre_x.permute(0, 2, 1)  # (batch, hidden_size, 1)

        return pre_x

    def search(self, input):
        """
        :param input: tensor, (1, seq_len, embed_size)
        :return:
        """
        m = input.shape[1]
        input = input.permute(1, 0, 2)
        x, (h_n, c_n) = self.lstm(input)   # (seq_len, 1, hidden_size)
        x = x.squeeze(1) # (seq_len, hidden_size)
        x = F.softmax(x, dim=1)

        pre_max = torch.zeros(x.shape[0], x.shape[1]).to(self.device)  # (seq_len, hidden_szie)
        # first 'start' to 'x1'
        pre_x = x[0, :].reshape(-1, 1)
        for i in range(1, m):
            Z = pre_x + self.transition + x[i, :]   # (hidden_size, hidden_size)
            pre_max[i, :] = torch.argmax(Z, dim=0)
            pre_x = torch.max(Z, dim=0)[0]

        last = torch.argmax(pre_x).item()
        pr = [last]
        for i in range(1, m):
            last = int(pre_max[m-i, last].item())
            pr.append(last)

        pr = pr[::-1]
        return pr

