### language model with RNN

1.对于输出预测，先不对softmax进行优化。
后面有机会再写分层softmax。

2.生成句子，不使用beam search，哪个大选哪个。

3.模型评测指标：困惑度(Hp) = 2^(损失函数)。

mmp，Hp overflow，光看loss感觉还行，但是生成的句子都是<pad><pad><pad>
现在<pad>不输出了，因为什么也不输出了

数据集中的每个中文分词大部分只出现一次左右，也就是是说理论上如果模型训练好，后面的词基本上是固定的。
需要额外添加新的古诗词？

fuck fuck fuck

给不同的类别添加不同的权重，<END>，<PAD>添加较小的权重
此时模型基本不收敛，之前模型收敛的基本上是关于<END>和<PAD>的？

训练一个词向量试试
都木有用

当我去掉所有的<START><END><PAD>的时候，模型收敛了。
其他词的频率实在是太低了，对于每个句子最后一个词总会是<END>，存在大量的输入词，预测输出词<END>，
导致训练之后无论输入什么词都会输出<END>，我猜是这样的。
调大embed_size和hidden_size加速收敛。

train_data收敛了，但是dev_data没什么变化，说明过拟合？


参考：
* https://github.com/Alic-yuan/nlp-beginner-finish/tree/master/task5
* cs224n hw