# TensorFlow 2.0 GPT

Structure is as same as [imcaspar/gpt2-ml](https://github.com/imcaspar/gpt2-ml), not official work, because I want to read the pretrained Chinese model. ([Chinese pretrained model](https://drive.google.com/file/d/1mT_qCQg4AWnAXTwKfsyyRWCRpgPrBJS3))

Still some difference between the three GPT code which refer, check the call function of `Transformer` class in the `model.py` for more detail.

Hint, that [imcaspar/gpt2-ml](https://github.com/imcaspar/gpt2-ml) 's GPT seems not the GPT-2, but GPT-1, even they write GPT-2 in their README. GPT-2 move layer-norm to the top of transformer block, and add an additional layer-norm after the final transformer block, check [here](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#openai-gpt-2) for more detail.

Paper:

- GPT-2: [https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)


References:

- Official repo (with TensorFlow 1.x) [openai/gpt-2](https://github.com/openai/gpt-2)
- [imcaspar/gpt2-ml](https://github.com/imcaspar/gpt2-ml)
- [karpathy/minGPT](https://github.com/karpathy/minGPT)
- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#openai-gpt-2)
