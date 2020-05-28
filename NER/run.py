from utils import loadData
import torch
import time
import torch.nn as nn
import torch.optim as optim
from rnn_model import RNN
from Vocab import Vocab
from utils import loadEmbeddings
from utils import batch_iter
from utils import loss_curve
from utils import data_split

def createVocabList(data):
    print("create vocab list")
    vocabList = {word:1 for sents in data for word in sents}
    vocabList = sorted(vocabList.keys())
    return vocabList

def train(train_data, train_labels, test_data, test_labels,
          model, optimizer, device, epoch, batch, params_path, corate=-1):
    model.train()

    log_every = 100
    save_iter = 1200

    embeddings = loadEmbeddings(model.vocab, model.embed_size,
                                './data/word2vec.6B.100d.txt')

    loss_data = []
    hint_correct_rate = [corate]
    s = time.time()

    for times in range(epoch):
        train_iter = 0

        for train_x, train_y in batch_iter(train_data, train_labels, batch):
            train_x = model.vocab.to_input_tensor(train_x)
            train_x = embeddings(train_x).to(device)
            train_y = torch.tensor(train_y, device=device)

            optimizer.zero_grad()
            loss = model(train_x, train_y)
            loss.backward()
            optimizer.step()

            if train_iter == 0:
                loss_data.append(loss.item())
            if train_iter % log_every == 0:
                e = time.time()
                print("the {} epoch, the {} iter, loss is : {} [time is : {}]".format(times,
                                                                                      train_iter, loss.item(), (e-s)/60.))

            if train_iter == save_iter:
                # auto save params
                correct_rate = test(test_data, test_labels, model, device, training=1)
                if correct_rate > max(hint_correct_rate):
                    checkpoint = {
                        'model_dict': model.state_dict(),
                        'corate': correct_rate
                    }
                    torch.save(checkpoint, params_path)
                    print('save params')
                else:
                    print('not save params')
                hint_correct_rate.append(correct_rate)

            train_iter += 1

    loss_curve(loss_data)
    print(hint_correct_rate)
    # load best model
    state = torch.load(params_path)
    model.load_state_dict(state['model_dict'])

def test(test_data, labels, model, device, batch=1, training=0):
    model.eval()

    embeddings = loadEmbeddings(model.vocab, model.embed_size,
                                './data/word2vec.6B.100d.txt')

    count, correct_count = 0, 0
    with torch.no_grad():
        for test_x, test_y in batch_iter(test_data, labels, batch):
            test_x = model.vocab.to_input_tensor(test_x)
            test_x = embeddings(test_x).to(device)

            output = model.search(test_x)

            test_y = test_y[0]
            for i in range(len(test_y)):
                count += 1
                if test_y[i] == output[i]:
                    correct_count += 1

        correct_rate = 1.*correct_count/count
        print('the corrent rate is : ', correct_rate)

    if training:
        model.train()
    return correct_rate


def main():
    name2id = {'START':0, 'I-MISC':1, 'B-MISC':2, 'I-LOC':3, 'B-LOC':4,
               'I-ORG':5, 'B-ORG':6, 'I-PER':7, 'O':8, 'END':9}

    train_data_path = './data/conll2003/eng.train'
    params_path = './data/rnn_params.pth'

    data, labels = loadData(train_data_path)
    labels = [[name2id[name] for name in sents] for sents in labels]

    vocabList = createVocabList(data)

    train_x, test_x, train_y, test_y = data_split(data, labels, 0.1, 42)

    torch.manual_seed(64)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch = 1
    epoch = 4
    embed_size = 100
    hidden_size = 9
    flag_load_model = 1
    corate = -1

    model = RNN(embed_size, hidden_size, vocabList, device).to(device)
    if flag_load_model:
        checkpoint = torch.load(params_path)
        model.load_state_dict(checkpoint['model_dict'])
        corate = checkpoint['corate']

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    for key, value in model.named_parameters():
        print(key, value.shape)

    train(train_x, train_y, test_x, test_y,
          model, optimizer, device, epoch, batch, params_path, corate=corate)

    print(model.transition)

if __name__ == "__main__":
    main()