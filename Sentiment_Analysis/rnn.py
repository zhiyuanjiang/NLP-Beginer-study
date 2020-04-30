import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import sys
import time
sys.path.append('/home/zdf/fage/nlp-beginer')
sys.path.append('f:/NLP-Beginner-study')
from Sentiment_Analysis.utils import Vocab
from Sentiment_Analysis.pretendOF import Regularization
from Sentiment_Analysis.utils import loadWordEmbedding
from Sentiment_Analysis.utils import batch_iter
from Sentiment_Analysis.bayes import loadDataSet
from Sentiment_Analysis.bayes import data_split
from Sentiment_Analysis.bayes import createVocabList
from Sentiment_Analysis.utils import batch_iter_test
from Sentiment_Analysis.utils import loss_curve

class RNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocabList, device):
        super(RNN, self).__init__()

        self.vocab = Vocab(vocabList)
        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.model_embeddings = loadWordEmbedding(self.vocab)

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.classify = nn.Linear(2*hidden_size, 5)
        self.drop = nn.Dropout(0.8)
        self.activate = nn.ReLU()
        self.h_0 = None
        self.c_0 = None
        self.h_n = None
        self.c_n = None

    def forward(self, data):
        batch = len(data)
        x = self.vocab.to_input_tensor(data, self.device, -1)
        self.h_0 = torch.zeros(2, batch, self.hidden_size, device=self.device)
        self.c_0 = torch.zeros(2, batch, self.hidden_size, device=self.device)
        x = self.model_embeddings(x)

        x = x.permute(1, 0, 2)   # (seq_len, batch, input_size)
        # rnn, lstm
        x, (self.h_n, self.c_n) = self.lstm(x, (self.h_0, self.c_0)) # x: (seq_len, batch, num_direction*hidden_size)
        x = self.activate(x)
        x = x.permute(1, 2, 0) # x: (batch, num_direction*hidden_size, seq_len)
        seq_len = x.shape[0]
        # average pooling
        # x = 1./seq_len*torch.sum(x, dim=2)
        # max pooling
        x = torch.max(x, dim=2)[0]
        # dropout prevent overfitting
        x = self.drop(x)
        # fulling connection
        x = self.classify(x)
        return F.log_softmax(x, dim=1)

def loss_func(output, target, reduction='mean'):
    return F.nll_loss(output, target, reduction=reduction)

def train(model, device, train_data, labels, optimizer, epoch, batch, weight_decay=0.):

    log_every = 100
    if weight_decay > 0.:
        reg_loss = Regularization(model, batch, weight_decay, p=2)
    # set the model to training model
    model.train()
    loss_data = []
    for idx in range(epoch):
        train_iter = 0
        for data, target in batch_iter(train_data, labels, batch, True):
            target = torch.tensor(target)
            target = target.to(device)
            # clear gradient
            optimizer.zero_grad()
            # forward propagation
            output = model(data)
            # calculation loss
            if weight_decay <= 0.:
                loss = loss_func(output, target)
            else:
                loss = loss_func(output, target)+reg_loss(model)
            # backward propagation
            loss.backward()
            # update parameters
            optimizer.step()

            if train_iter == 0:
                loss_data.append(loss.item())
            if train_iter % log_every == 0:
                print("the {} epoch, the {} iter, loss is : {}".format(idx, train_iter, loss.item()))

            train_iter += 1

    loss_curve(loss_data)
    # torch.save(model.state_dict(), './data/rnn_params.pth')

def test(model, device, test_data, labels):
    model.eval()
    m = len(labels)
    batch_size = 200
    correct_count = 0

    for data, y in batch_iter(test_data, labels, batch_size):
        target = torch.tensor(y)
        target = target.to(device)
        output = model(data)
        predict = torch.argmax(output, dim=1)
        correct_count += torch.sum(torch.eq(target, predict))

    print('the correct rate is: ', 1.*correct_count.item()/m)

def kaggleTest(model, filePath):
    test_data, labels = loadDataSet('./data/test.tsv', 1)

    batch_size = 200
    cnt = len(test_data)
    # newline='' 不会产生空行
    with open(filePath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PhraseId', 'Sentiment'])

        number = 156061
        for data in batch_iter_test(test_data, batch_size):
            model.eval()
            output = model(data)
            predict = torch.argmax(output, dim=1)

            tid = [number + i for i in range(len(predict))]
            kaggle_data = list(zip(tid, predict.cpu().numpy().tolist()))

            number += len(predict)
            writer.writerows(kaggle_data)

    print("the amount of data is : ", cnt)

def main():
    print("rnn algorithm")
    train_data, labels = loadDataSet("./data/train.tsv")
    test_data, L = loadDataSet('./data/test.tsv', 1)

    train_x, test_x, train_y, test_y = data_split(train_data, labels, 0.1, 42)
    # 所有文件中最长的评论长度
    # max_sent_len = 56

    # 只使用训练样本中出现的词
    vocabListTrainData = createVocabList(train_data)
    # 使用测试样本出现的词
    vocabListTestData = createVocabList(test_data)
    # 使用词表中的所有词
    # vocabListGlove = createPreTrainVocab()
    vocabList = vocabListTrainData | vocabListTestData

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(64)

    device = torch.device("cuda" if use_cuda else "cpu")

    batch = 64
    epoch = 32
    embed_size = 100
    hidden_size = 50

    model = RNN(embed_size, hidden_size, vocabList, device).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    flag = 0
    if flag == 0:
        s = time.time()
        train(model, device, train_x, train_y, optimizer, epoch, batch, 0.2)
        e = time.time()
        print("train time is : ", (e-s)/60.)
    else:
        model.load_state_dict(torch.load('./data/cnn_params.pth'))

    test(model, device, train_x, train_y)
    test(model, device, test_x, test_y)

    kaggleTest(model, './data/kaggleData.csv')

if __name__ == "__main__":
    main()