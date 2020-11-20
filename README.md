# TensorFlow 2.x CPM-Generate

本Repo将模型转换为TensorFlow版本，原Repo [https://github.com/TsinghuaAI/CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)

原项目首页：[https://cpm.baai.ac.cn/](https://cpm.baai.ac.cn/)

原项目介绍文章：[https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw](https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw)

    如果你只想大概看一下结果，请直接打开`prediction_v2.ipynb`文件预览

## 感谢智源研究院的工作！

Q：为什么你要把源代码转换为TensorFlow版本？

A：1. 我装不上Nvidia apex；2. 按照原Repo说明需要两张显卡的主机运行，我也没有； 3. TensorFlow模型很方便。

^_^ 如果你喜欢我的工作，请给智源研究院的原Repo打星，如果能顺便给我的Repo也打一个就更好了。

## 使用方法

1. Clone本Repo

2. 下载模型：

打开下面分享下载CPM目录的`cpm-lm-tf2_v2`


```
链接: https://pan.baidu.com/s/1tjbWty2hkbmtCrvV9Qh_SQ  密码: n0nt
--来自百度网盘超级会员V7的分享

or GDrive：

https://drive.google.com/drive/folders/1b2sF5sBuR_9zsT8UUijdsAcmFaMZJlpX?usp=sharing
```

    另一个目录`cpm-lm-tf2-fp16`是fp16版本的模型，但是除非你有显卡，并且确认支持float16，并且确认正确安装CUDA，否则`请使用非fp16`的版本，因为在不支持float16的设备上，会非常慢！

下载`cpm-lm-tf2`到Clone好的Repo目录，结构大概是这样：

```
- cpm-tf2
  - cpm-lm-tf2_v2  （从网盘下载好的TF2版本模型）
    - assets
    - saved_model.pb
    - variables
  - CPM-Generate
    - bpe_3w_new （词表所在目录）
  - prediction_v2.ipynb  （预测demo，主程序）
  - gpt2_tokenizer.py  （分词文件，这个里面引入了jieba，和huggingface那一系列的不能简单互换）
```

    运行所需的代码其实就大概以上的几个文件和目录就够了，其他的主要是模型转换等代码

3. 安装依赖

```
# 依赖：
pip install sentencepiece jieba regex tensorflow tensorflow-hub
```

4. 参考`prediction_v2.ipynb`中的代码运行


其中所需大概代码就这么几行：

```python
import tensorflow_hub as hub
import tensorflow as tf

from gpt2_tokenizer import GPT2Tokenizer

tokenizer = GPT2Tokenizer(
    'CPM-Generate/bpe_3w_new/vocab.json',
    'CPM-Generate/bpe_3w_new/merges.txt',
    model_file='CPM-Generate/bpe_3w_new/chinese_vocab.model')

gpt = hub.load('./cpm-lm-tf2_v2/')


def sample(tokenizer, gpt, sentence, number=1, length=20, top_p=0.9, temperature=0.9):
    """
    numbert: 输出句子个数
    length: 输出最大长度
    top_p: token的概率排在这以上才有效
    temperature: 温度
    """
    inputs = tf.constant([tokenizer.encode(sentence)] * number, dtype=tf.int64)
    length = tf.constant(length, dtype=tf.int64)
    ret = gpt.signatures['serving_default'](
        inp=inputs,
        length=length,
        top_p=tf.constant(top_p, tf.float32),
        temperature=tf.constant(temperature, tf.float32)
    )['output_0']
    return [
        tokenizer.decode(s).replace(' ', '')
        for s in ret.numpy()
    ]


ret = sample(tokenizer, gpt, '默写英文：\n狗dog\n猫cat\n鸟', 3, 10, top_p=0.9, temperature=0.9)
for x in ret:
    print(x)
    print('-' * 20)
```

# md5

cpm-lm-tf2_v2

```
e075f997f257cbd0a5d9cbb972322d46 saved_model.pb
0b7c8b55e1061a0561a5242005015295 variables/variables.data-00000-of-00001
86b97441fad0a3f2a616bf55f22d866a variables/variables.index
```

cpm-lm-tf2-fp16

```
6958f42f74dd8e4b45ef60628b80da57 saved_model.pb
19bb9afa028a05ea03d4fe52e47d3c48 variables/variables.data-00000-of-00001
462fc087aa454887158f353fa0316801 variables/variables.index
```

## 一些额外的闲聊

### 模型的转换参考

模型的具体转换代码在`load_pytorch.ipynb`文件中，希望有类似torch to tensorflow的同学可以参考

### 原模型的一些细节

模型的训练基础是英伟达的[https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)，这大概算是一个英伟达魔改的PyTorch上的高级API，论文在[https://arxiv.org/pdf/1909.08053.pdf](https://arxiv.org/pdf/1909.08053.pdf)。

这个应该主要是为了能把一个很大的模型在很多张显卡上更好的并行训练而设计的，原模型分了两个文件，也提到了需要两张显卡，应该是在每张显卡上分别载入这两个文件。

这两个文件中大概各有一半的模型参数，有些层，例如全连接层（Dense）的参数会平均到两个模型中。

比如一个256到256的Dense层，按道理来说有一个256x256的kernel和一个256的bias，平分之后会在每个文件分别有一个128x256的kernel和一个128的bias。

因为有些层无法平分，例如LayerNormalization层，所以是在两个文件中有重复的。

### fp32和fp16

fp16在笔者的CPU上几乎和龟速一样（Macbook Pro 2020），比fp32的慢了好多倍。

猜测这应该是由于现代cpu上实际上不具备物理的fp16运算器导致的，也就是每次进行fp16的前后其实是把fp16转换为了32再运行的，所以非常浪费CPU。

fp16的模型相比fp32的其实是有一些区别的，主要是原来的attention mask的问题，因为1e10这个数字在fp32上是合法的，但是在fp16上是inf，所以笔者把这部分mask的1e10的超参改为了1e4，才跑起来fp16的模型。

### TensorFlow版本和原版本的区别

道理来讲应该没有什么太大区别，而且也载入了原来的参数，不过毕竟还是有GPU -> CPU，PyTorch -> TensorFlow这样的转换，所以可能和原模型结果有一定出入，不过笔者估计这个出入不会很大，顶多1%左右。

## 引用

参考原Repo

```
@article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Cao, Jiannan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
```