
class Config(object):

    def __init__(self):

        self.batch = 1
        self.epoch = 16
        self.embed_size = 200
        self.hidden_size = 200
        # 是否加载模型参数, 0: 不加载
        self.flag_load_model = 1
        self.train_data_path = './data/poetryFromTang.txt'
        self.params_path = './data/rnn_params.pth'

        self.lr = 0.1
        self.momentum = 0.9