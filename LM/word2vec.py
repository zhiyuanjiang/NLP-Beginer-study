import logging
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from utils import loadRawData
from utils import processRawData

def train_word2vec(sentences):
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences, sg=1, size=100, window=10, min_count=1, workers=4, iter=16)
    model.wv.save_word2vec_format("./data/w2v100d.bin", binary=False)

def main():
    train_data_path = './data/poetryFromTang.txt'
    rawdata = loadRawData(train_data_path)
    data = processRawData(rawdata)
    train_word2vec(data)

    # model = KeyedVectors.load_word2vec_format('./data/w2v50d.bin', binary=True)
    # print(model['645764'])
    # print(model.most_similar('645764'))
    # print("test")


if __name__ == "__main__":
    main()