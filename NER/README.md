### NER

problem 1:
loss是nan，计算exp(Z)的时候发生了溢出。于是我想着对Z进行裁剪。
结果出现了loss为负的现象。
loss = Ps-Pr，理论上是不可能的，因为Pr不可能比Ps大，但因为我对Z进行了裁剪，如果句子一长，就变得有可能了。

solution:
exp(Z)-max(Z), 最后再补上去，balabala....


参考：
* https://github.com/createmomo/CRF-Layer-on-the-Top-of-BiLSTM
* https://zhuanlan.zhihu.com/p/119254570


