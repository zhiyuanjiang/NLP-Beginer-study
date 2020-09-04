import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
# sys.path.append('/home/zdf/fage/nlp-beginer')
from utils import loadWordEmbedding
from utils import pad_sents
from utils import Vocab
from utils import batch_iter
from utils import createPreTrainVocab
from utils import loadDataSet
from utils import data_split
from utils import createVocabList
from pretendOF import Regularization
import utils
import csv

class CNN(nn.Module):
    """
    一个简单卷积神经网络。
    进行情感分类。
    """
    def __init__(self, embed_size, max_sent_len, vocabList, device):

        super(CNN, self).__init__()

        self.max_sent_len = max_sent_len
        self.vocab = Vocab(vocabList)
        self.model_embeddings = loadWordEmbedding(self.vocab)

        self.feature = nn.Sequential(
            nn.Conv1d(embed_size, 64, 6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 16, 6),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*23, 120),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

        self.device = device

    def forward(self, data):
        # 批量数据进行转换，考虑到内存问题
        x = self.vocab.to_input_tensor(data, self.device, self.max_sent_len)
        x = self.model_embeddings(x)
        x = x.permute(0, 2, 1)

        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


def loss_func(output, target, reduction='mean'):
    return F.nll_loss(output, target, reduction=reduction)


def train(model, device, train_data, labels, optimizer, epoch, batch, weight_decay=0.1):

    log_every = 100
    if weight_decay > 0.:
        reg_loss = Regularization(model, batch, weight_decay, p=2)
    # set the model to training model
    model.train()
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

            if train_iter % log_every == 0:
                print("the {} epoch, the {} iter, loss is : {}".format(idx, train_iter, loss.item()))

            train_iter += 1

    torch.save(model.state_dict(), './data/cnn_params.pth')


def test(model, device, test_data, labels):
    model.eval()
    m = len(labels)
    target = torch.tensor(labels)
    target = target.to(device)
    output = model(test_data)
    predict = torch.argmax(output, dim=1)
    correct_count = torch.sum(torch.eq(target, predict))
    print('the correct rate is: ', 1.*correct_count.item()/m)


def kaggleTest(model, filePath):
    test_data, labels = loadDataSet('./data/test.tsv', 1)
    model.eval()
    output = model(test_data)
    predict = torch.argmax(output, dim=1)

    tid = [156061 + i for i in range(len(predict))]
    kaggle_data = list(zip(tid, predict.numpy().tolist()))

    print('the test data count is : ', len(predict))
    # print(kaggle_data)
    # newline='', 就不会产生空行
    with open(filePath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PhraseId', 'Sentiment'])
        writer.writerows(kaggle_data)

if __name__ == "__main__":
    print("cnn algrithm")

    train_data, labels = loadDataSet("./data/train.tsv")
    test_data, _ = loadDataSet('./data/test.tsv', 1)

    train_x, test_x, train_y, test_y = data_split(train_data, labels, 0.1, 42)
    # 所有文件中最长的评论长度
    max_sent_len = 56

    # 只使用训练样本中出现的词
    vocabListTrainData = createVocabList(train_data)
    # 使用测试样本出现的词
    vocabListTestData = createVocabList(test_data)
    # 使用词表中的所有词
    # vocabListGlove = createPreTrainVocab()
    vocabList = vocabListTrainData | vocabListTestData
    vocabList = sorted(vocabList)

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(64)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = CNN(100, max_sent_len, vocabList, device).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    flag = 0
    if flag == 0:
        train(model, device, train_x, train_y, optimizer, 15, 64)
    else:
        # 貌似这个要放在优化器前面
        model.load_state_dict(torch.load('./data/cnn_params.pth'))

    test(model, device, train_x, train_y)
    test(model, device, test_x, test_y)

    kaggleTest(model, './data/kaggleData.csv')
