# TensorFlow 2.x CPM-Generate

本Repo将模型转换为TensorFlow版本，原Repo [https://github.com/TsinghuaAI/CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)

原项目首页：[https://cpm.baai.ac.cn/](https://cpm.baai.ac.cn/)

原项目介绍文章：[https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw](https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw)

    如果你只想大概看一下结果，请直接打开`prediction_large.ipynb`文件预览

## 感谢智源研究院的工作！

^_^ 如果你喜欢我的工作，请给智源研究院的原Repo打星，如果能顺便给我的Repo也打一个就更好了。

## 使用方法

**HINT：请主要使用TensorFlow 2.3.0以上版本测试，其他版本可能出兼容性问题，本项目以学习为主，以后也不会做太多兼容性优化请见谅**

1. Clone本Repo

2. 下载模型：

```

百度网盘，下载大模型 cpm-large-tf2
链接: https://pan.baidu.com/s/1gup1qhojFr4jC_a70tlmgw  密码: 5lba

百度网盘，下载小模型 cpm-distill-tf2
链接: https://pan.baidu.com/s/10vhjVRX2tWbX2892ulLemg  密码: dsvv

or GDrive：

大模型
https://drive.google.com/drive/folders/1XGy2B6QSf1k0SOtVF13gcekLvdtpPkmq?usp=sharing

小模型
https://drive.google.com/drive/folders/13wTPVMEslAx8Xl0Sfw59BbSUR-BowHgW?usp=sharing

```

下载`cpm-large-tf2`，或者下载小模型`cpm-distill-tf2`，自己选一个就好，结果上肯定是大模型更好，具体区别请自己对比`prediction_large.ipynb`和`prediction_distill.ipynb`中的结果

下载到Clone好的Repo目录，结构大概是这样：

```
cpm-lm-tf2/
...cpm-large-tf2/  （从网盘下载好的TF2版本大模型）
......assets
......saved_model.pb
......variables
...cpm-distill-tf2/  （从网盘下载好的TF2版本小模型，大小两个下载其中一个就好）
......assets
......saved_model.pb
......variables
...CPM-Generate/
......bpe_3w_new/ （词表所在目录）
...prediction_large.ipynb  （预测大模型的demo主程序，下载了大模型就运行这个）
...prediction_distill.ipynb  （预测小模型的demo主程序，下载了小模型就运行这个）
...gpt2_tokenizer.py  （分词文件，这个里面引入了jieba，和huggingface那一系列的不能简单互换）
```

    运行所需的代码其实就大概以上的几个文件和目录就够了，其他的主要是模型转换等代码

3. 安装依赖

```
# 依赖：
pip install sentencepiece jieba regex tensorflow tensorflow-hub
```

4. 参考`prediction_large.ipynb`中的代码运行，或参考小模型的`perdiction_distill.ipynb`

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
