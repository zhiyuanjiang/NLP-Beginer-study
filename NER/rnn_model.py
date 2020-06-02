import torch
import torch.nn as nn
import torch.nn.functional as F
from Vocab import Vocab

class RNN(nn.Module):
    """
    from https://github.com/createmomo/CRF-Layer-on-the-Top-of-BiLSTM
    classify_size - the number of classification
    """

    def __init__(self, embed_size, hidden_size, n_label, vocabList, device):
        super(RNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = Vocab(vocabList)
        self.device = device
        self.n_label = n_label  # 'start' and 'end' should be include

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        # the params to be add model, transition matrix
        self.transition = nn.Parameter(torch.rand(n_label, n_label)*0.1)
        self.classify = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(32, n_label)
        )


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

        m = inputs.shape[0]  # seq_len
        n = inputs.shape[1]  # batch
        x, (h_n, c_n) = self.lstm(inputs)  # (seq_len, batch, hidden_size)
        x = self.classify(x.permute(1, 0, 2))   # (batch, seq_len, n_label)
        x = F.softmax(x, dim=2)

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
        :param H: (batch, seq_len, n_label)
        :return:
            scores - tensor, (batch, )
        """
        m = H.shape[1]  # seq_len
        transition = F.softmax(self.transition, dim=1)
        # first, calculate the score of 'start' to 'x1'
        pre_x = H[:, 0, :] + transition[0, :]   # (batch, n_label)
        pre_x = pre_x.unsqueeze(1)  # (batch, 1, n_label)
        pre_x = pre_x.permute(0, 2, 1)  # (batch, n_label, 1)

        for i in range(1, m):
            Z = pre_x + transition + H[:, i, :].unsqueeze(1)   # (batch, n_label, n_label)
            # to avoid the score overflow
            pre_x = self.log_sum_exp(Z)

        # finally, calculate the score of 'xn' to 'end'
        scores = pre_x + transition[-1, :].reshape(-1, 1)  # (batch, n_label, 1)
        scores = self.log_sum_exp(scores)
        scores = scores.squeeze()

        return scores

    def log_sum_exp(self, Z):
        """
        :param Z: (batch, hidden_size, hidden_size)
        :return:
        """
        max_val = torch.max(Z, dim=1, keepdim=True)[0]  # (batch, 1, n_label)
        pre_x = torch.sum(torch.exp(Z-max_val), dim=1, keepdim=True)  # (batch, 1, n_label)
        pre_x = torch.log(pre_x) + max_val
        pre_x = pre_x.permute(0, 2, 1)  # (batch, n_label, 1)

        return pre_x

    def search(self, input):
        """
        :param input: tensor, (1, seq_len, embed_size)
        :return:
        """
        m = input.shape[1]
        input = input.permute(1, 0, 2)
        x, (h_n, c_n) = self.lstm(input)   # (seq_len, 1, hidden_size)
        x = self.classify(x.permute(1, 0, 2))

        x = x.squeeze(0) # (seq_len, n_label)
        x = F.softmax(x, dim=1)

        pre_max = torch.zeros(x.shape[0], x.shape[1]).to(self.device)  # (seq_len, n_label)
        # first 'start' to 'x1'
        pre_x = x[0, :].reshape(-1, 1)
        for i in range(1, m):
            Z = pre_x + self.transition + x[i, :]   # (n_label, n_label)
            pre_max[i, :] = torch.argmax(Z, dim=0)
            pre_x = torch.max(Z, dim=0)[0]

        last = torch.argmax(pre_x).item()
        pr = [last]
        for i in range(1, m):
            last = int(pre_max[m-i, last].item())
            pr.append(last)

        pr = pr[::-1]
        return pr

