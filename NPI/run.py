import time
from tqdm import tqdm
import pandas as pd
import torch
import csv
import string
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append('/home/zdf/fage/nlp-beginer')
from NPI.rnn_model import RNN
from NPI.utils import loss_curve
from NPI.utils import batch_iter

"""
# neutral: 0
# contradiction: 1
# entailment: 2
"""

def loadData(filePath):
    """
    加载数据
    :param filePath:
    :return:
         labels - list[list[int]]
         premise - list[list[str]], premise data, 没有对标点符号进行处理
         hypothesis - list[list[str]]
    """
    s = time.time()
    data = pd.read_table(filePath)
    # 如果sentence2那一列中某个单元存在NAN，那么去掉那个单元对应的某一行
    data = data.dropna(subset={'sentence2'}, how='any')
    # 选取gold_label列中不包含'-'的行
    data = data[~data['gold_label'].isin(['-'])]
    label_dict = {'neutral':0, 'contradiction':1, 'entailment':2}
    labels = list(map(lambda x: label_dict[x], data['gold_label']))
    table = str.maketrans('', '', string.punctuation)
    premis = list(map(lambda x: x.lower().translate(table).split(), data['sentence1']))
    hypothesis = list(map(lambda x: x.lower().translate(table).split(), data['sentence2']))
    e = time.time()
    print('load data time is : ', e-s)
    return labels, premis, hypothesis

def createVocabList(dataSet):
    """
    使用dataSet创建一个词汇表
    :param dataSet: list[list[str]]
    :return:
    """
    vocabList = set() # 词表
    print('create vocab list')
    for id, data in tqdm(enumerate(dataSet)):
        vocabList = vocabList|set(data)
    # return list(vocabList)
    return vocabList

def loss_func(output, target, reduction='mean'):
    return F.nll_loss(output, target, reduction=reduction)

def train(premise_data, hypothesis_data, labels, model, optimizer, epoch, batch, device):
    model.train()
    loss_data = []
    log_every = 100
    for idx in range(epoch):
        train_iter = 0
        for pdata, hdata, target in batch_iter(premise_data, hypothesis_data, labels, batch, True):
            target = torch.tensor(target)
            target = target.to(device)
            # clear gradient
            optimizer.zero_grad()
            # forward propagation
            output = model(pdata, hdata)
            # calculation loss
            loss = loss_func(output, target)
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
    torch.save(model.state_dict(), './data/rnn_params.pth')

def test(premise_data, hypothesis_data, labels, model, device, batch):
    model.eval()
    m = len(labels)
    correct_count = 0

    for pdata, hdata, target in batch_iter(premise_data, hypothesis_data, labels, batch):
        target = torch.tensor(target)
        target = target.to(device)
        output = model(pdata, hdata)
        predict = torch.argmax(output, dim=1)
        correct_count += torch.sum(torch.eq(target, predict))

    print('the correct rate is: ', 1.*correct_count.item()/m)

def saveVocabList(filePath, vocabList):
    with open(filePath, 'w', newline='') as f:
        writer = csv.writer(f)
        vocabList = [[it] for it in vocabList]
        writer.writerows(vocabList)

def readVocabList(filePath):
    vocabList = []
    with open(filePath, 'r') as f:
        data = csv.reader(f)
        for it in data:
            vocabList.extend(it)
    return vocabList

def main():
    print('NPI problem')
    train_data_path = './data/snli_1.0/snli_1.0_train.txt'
    dev_data_path = './data/snli_1.0/snli_1.0_dev.txt'
    vocab_path = './data/vocabList.txt'

    labels, premise, hypothesis = loadData(train_data_path)
    dev_labels, dev_premise, dev_hypothesis = loadData(dev_data_path)

    # vocabList = createVocabList(premise) | createVocabList(hypothesis)
    # saveVocabList(vocab_path, vocabList)

    vocabList = readVocabList(vocab_path)
    print(len(vocabList))

    torch.manual_seed(64)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch = 64
    epoch = 48
    embed_size = 100
    hidden_size = 100

    model = RNN(embed_size, hidden_size, vocabList, device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    s = time.time()
    train(premise, hypothesis, labels, model, optimizer, epoch, batch, device)
    e = time.time()
    print('the train time is : ', 1.*(e-s)/60)
    # model.load_state_dict(torch.load('./data/rnn_params.pth'))
    dev_labels, dev_premise, dev_hypothesis = loadData(dev_data_path)

    test(premise, hypothesis, labels, model, device, batch)
    test(dev_premise, dev_hypothesis, dev_labels, model, device, batch)

if __name__ == '__main__':
    main()