import torch
import torch.nn as nn
import torch.nn.functional as F
from Vocab import  Vocab

class PoetryModel(nn.Module):
    """
    language model
    """
    def __init__(self, embed_size, hidden_size, vocabList, device):
        super(PoetryModel, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.device = device
        self.vocab = Vocab(vocabList)

        self.embeddings = nn.Embedding(self.vocab.len, embed_size)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.classify = nn.Linear(hidden_size, self.vocab.len)

    def forward(self, inputs):
        """
        :param inputs: tensor (batch, seq_len)
        :return:
        """
        x = self.embeddings(inputs) # tensor (batch, seq_len, embed_size)

        m = x.shape[0]
        hidden_state = []
        h_i = torch.zeros(m, self.hidden_size).to(self.device)
        c_i = torch.zeros(m, self.hidden_size).to(self.device)
        for x_i in torch.split(x, 1, dim=1):
            x_i = x_i.squeeze(1)
            h_i, c_i = self.lstm(x_i, (h_i, c_i))    # (batch, hidden_size)
            hidden_state.append(h_i)

        hidden_state = torch.stack(hidden_state, dim=1)  # (batch, seq_len, hidden_size)
        y = self.classify(hidden_state)
        y = y.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        y = F.log_softmax(y, dim=1)

        return y

    def generate_poetry(self, max_len):

        x_i = self.vocab.start_token_id
        # x_i = 8
        x_i = self.embeddings(torch.tensor([x_i]))
        h_i = torch.zeros(1, self.hidden_size).to(self.device)
        c_i = torch.zeros(1, self.hidden_size).to(self.device)
        output = []

        for idx in range(max_len):
            h_i, c_i = self.lstm(x_i, (h_i, c_i))
            o_i = self.classify(h_i)
            t = torch.argmax(o_i, dim=1)
            # if t.item() == self.vocab.end_token_id:
            #     break
            output.append(t.item())
            x_i = self.embeddings(torch.tensor([t.item()]))

        output = [self.vocab.id2word[t] for t in output]
        print(output)
        sent = ""
        for s in output:
            sent = sent+s
        print(sent)
        return output
