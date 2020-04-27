kaggle：Sentiment Analysis on Movie Reviews

https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/overview

参考：
* Convolutional Neural Networks for Sentence Classification

1.bayes algorithm

使用词袋模型（训练样本构成的词汇表大概在1w左右）， 感觉效果不错。（0.592）

2.logistic regression

同样使用词袋模型（一个向量有点大），效果不理想。（0.518）

这里再用词向量试一下？

3.cnn

* 单写了一层卷积层和一个最大池化层，使用glove，效果不错。（0.597）

* 使用复杂一点的模型结构，性能得到一丢丢提升，但有一点点过拟合，不管怎么使用dropout，
L2正则化，效果都不理想。（0.615）

* 使用正则化后，模型的下降速度变慢了，loss的波动也变得平缓一些了，但没有什么用。

* 当更新参数时，对词向量进行微调，性能又有了一丢丢提升，过拟合更加严重。（0.629）

* 上面都只是使用了训练数据中的单词作为词汇表。对test data进行处理后发现，存在近3000个生词，近2w左右的样本
存在至少包含一个生词（6w左右的数据）。使用glove作为词汇表中的所有词，只有近500的生词，2000左右的样本至少包含
一个生词，大部分生词都是组合词，xxx-xxx。接下来使用训练样本和glove中的所有词作为词汇表。理论上性能还是要有
提升吧。（速度慢了不知道多少，太难了）

貌似对模型没有影响，莫非那些生词都是非关键词？

* 又xjb弄了几层dropout，搞了个L2正则化，epoch=15（减少了训练次数），竟然避免了过拟合，准确度达到了0.637。
哈哈，调不动了。