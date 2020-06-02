
class Config(object):

    def __init__(self):

        self.batch = 16
        self.epoch = 8
        self.embed_size = 50
        self.hidden_size = 50
        # 是否加载模型参数, 0: 不加载
        self.flag_load_model = 0
        self.train_data_path = './data/poetryFromTang.txt'
        self.params_path = './data/rnn_params.pth'

        self.lr = 0.1
        self.momentum = 0.9