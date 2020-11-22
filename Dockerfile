FROM python:3.6

RUN pip install sentencepiece jieba regex tensorflow tensorflow-hub -i https://pypi.tuna.tsinghua.edu.cn/simple

