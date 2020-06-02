from utils import loadRawData
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rnn_model import PoetryModel
from Vocab import Vocab
from utils import batch_iter
from utils import loss_curve
from utils import data_split
from utils import processRawData
from Config import Config
import math

def createVocabList(data):
    print("create vocab list")
    vocabList = {word:1 for sents in data for word in sents}
    vocabList = sorted(vocabList.keys())
    return vocabList

def loss_function(output, target, reduction='mean'):
    return F.nll_loss(output, target, reduction=reduction)

def train(train_data, test_data, model, optimizer, device, config, Hp):
    model.train()

    log_every = 3
    save_iter = 2

    loss_data = []
    s = time.time()

    hint_hp = [Hp]

    for times in range(config.epoch):
        train_iter = 0

        for ndata in batch_iter(train_data, config.batch):

            ndata = [[model.vocab.start_token]+sent+[model.vocab.end_token] for sent in ndata]
            ndata = model.vocab.to_input_tensor(ndata).to(device)
            train_x = ndata[:, :-1]
            train_y = ndata[:, 1:]
            optimizer.zero_grad()
            output = model(train_x)

            loss = loss_function(output, train_y)
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
                Hp = test(test_data, model, device, config, training=1)
                if Hp < min(hint_hp):
                    checkpoint = {
                        'model_dict': model.state_dict(),
                        'Hp': Hp
                    }
                    torch.save(checkpoint, config.params_path)
                    print('save params')
                else:
                    print('not save params')
                hint_hp.append(Hp)

            train_iter += 1

    loss_curve(loss_data)
    print(hint_hp)
    # load best model
    state = torch.load(config.params_path)
    model.load_state_dict(state['model_dict'])

def test(test_data, model, device, config, training=0):
    # use the perplexity as a evaluation indicator
    model.eval()
    count, correct_count = 0, 0
    Hp = 0.
    with torch.no_grad():
        for ndata in batch_iter(test_data, config.batch):

            ndata = [[model.vocab.start_token] + sent + [model.vocab.end_token] for sent in ndata]
            ndata = model.vocab.to_input_tensor(ndata).to(device)
            test_x = ndata[:, :-1]
            test_y = ndata[:, 1:]
            output = model(test_x)

            Hp = Hp + loss_function(output, test_y, reduction='sum').item()


        m = len(test_data)
        Hp = 1./m*Hp
        # Hp = math.pow(2, 1./m*Hp)
        print('the perplexity is : ', Hp)

    if training:
        model.train()
    return Hp


def main():
    config = Config()

    rawdata = loadRawData(config.train_data_path)
    data = processRawData(rawdata)
    vocabList = createVocabList(data)

    train_x, test_x = data_split(data, 0.1, 42)

    torch.manual_seed(64)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = PoetryModel(config.embed_size, config.hidden_size, vocabList, device).to(device)

    hp = float('inf')
    if config.flag_load_model:
        checkpoint = torch.load(config.params_path)
        model.load_state_dict(checkpoint['model_dict'])
        hp = checkpoint['Hp']

    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.lr)

    for key, value in model.named_parameters():
        print(key, value.shape)

    train(train_x, test_x, model, optimizer, device, config, Hp=hp)
    model.generate_poetry(24)


if __name__ == "__main__":
    main()