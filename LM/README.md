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

参考：
* https://github.com/Alic-yuan/nlp-beginner-finish/tree/master/task5
* cs224n hw