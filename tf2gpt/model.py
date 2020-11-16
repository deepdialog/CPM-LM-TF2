"""
Paper:
- https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

Official repo:
- https://github.com/openai/gpt-2

Other reference repo:
- https://github.com/imcaspar/gpt2-ml
- https://github.com/karpathy/minGPT

"""


import math
import numpy as np
import tensorflow as tf


@tf.function
def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the GELU activation applied.

    Way 1:
    https://github.com/tensorflow/addons/blob/81529ff7dd246f7575338b8cfe65784b0cc8a502/tensorflow_addons/activations/gelu.py#L81

    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), tf.keras.backend.floatx())))

    Way 2:
    https://github.com/openai/gpt-2/blob/0574c5708b094bfa0b0f6dfe3fd284d9a045acd9/src/model.py#L25
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))

    In my test case, way 1 is faster than way 2

    def gelu1(x):
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), tf.keras.backend.floatx())))

    def gelu2(x):
        return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))

    def test(f):
        x = tf.random.uniform((100, 100))
        for _ in range(1000):
            x += f(tf.random.uniform((100, 100)))
        return x

    %timeit test(gelu1)
    %timeit test(gelu2)

    1 loop, best of 3: 262 ms per loop
    1 loop, best of 3: 306 ms per loop
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), tf.keras.backend.floatx())))


def get_attention_mask(dim):
    """
    给一个斜对角矩阵，让decoder的每个元素只能看到自己前面的，例如
    dim = 3
    1 0 0 
    1 1 0
    1 1 1

    Another way:
    https://github.com/openai/gpt-2/blob/0574c5708b094bfa0b0f6dfe3fd284d9a045acd9/src/model.py#L58
    
    i = tf.range(dim)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + dim
    return tf.cast(m, dtype)

    """
    return tf.cast(
        tf.linalg.band_part(tf.ones((dim, dim)), -1, 0),
        tf.keras.backend.floatx())


def get_dense(units, name=None, stddev=0.02):
    return tf.keras.layers.Dense(
        units=units,
        name=name,
        kernel_initializer=tf.keras.initializers.random_normal(stddev=stddev),
        bias_initializer=tf.keras.initializers.constant(0.0)
    )


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_size,
                 num_attention_heads,
                 attention_dropout,
                 residual_dropout, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.num_attention_heads = num_attention_heads
        self.size_per_head = embedding_size // num_attention_heads
        self.embedding_size = embedding_size

        # B, L, L, N, S
        # batch, seq_len, seq_len, num_attention_heads, size_per_head
        self.query = get_dense(units=embedding_size, name="query_layer")
        self.key = get_dense(units=embedding_size, name="key_layer")
        self.value = get_dense(units=embedding_size, name="value_layer")
        self.attn_drop = tf.keras.layers.Dropout(attention_dropout)
        self.resid_drop = tf.keras.layers.Dropout(residual_dropout)
        self.proj = get_dense(units=embedding_size, name='context_projection_layer')

    def call(self, x, kv_cache=None, **kwargs):
        seq_len = tf.shape(x)[1]

        k = self.key(x)
        # [B, L, N * S] -> [B, L, N, S]
        k = tf.reshape(k, [-1, seq_len, self.num_attention_heads, self.size_per_head])
        # [B, L, N, S] -> [B, N, L, S]
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = self.value(x)
        v = tf.reshape(v, [-1, seq_len, self.num_attention_heads, self.size_per_head])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        q = self.query(x)
        q = tf.reshape(q, [-1, seq_len, self.num_attention_heads, self.size_per_head])
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        cached_kv = tf.stack([k, v], axis=1)
        if kv_cache is not None:
            pk, pv = tf.unstack(kv_cache, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        
        attn = tf.matmul(q, k, transpose_b=True)  # [B, N, L, S]
        attn = tf.multiply(attn, 1.0 / math.sqrt(float(self.size_per_head)))

        # [L, S]
        attention_mask = get_attention_mask(seq_len)
        # [1, 1, L, S]
        attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 0), 0)

        if tf.keras.backend.floatx() == tf.float16:
            neg = float(1e4) * (1.0 - tf.cast(attention_mask, tf.keras.backend.floatx()))
        else:
            neg = float(1e10) * (1.0 - tf.cast(attention_mask, tf.keras.backend.floatx()))
        # adding to softmax -> its like removing them entirely
        attn = attn * attention_mask - neg
        attn = tf.nn.softmax(attn)  # [B, N, L, S]
        attn = self.attn_drop(attn)

        y = tf.matmul(attn, v)
        # [B, N, L, S] -> [B, L, N, S]
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        y = tf.reshape(y, [-1, seq_len, self.embedding_size])
        y = self.resid_drop(self.proj(y))

        return y, cached_kv

    
class Transformer(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size,
                 num_attention_heads,
                 attention_dropout,
                 residual_dropout,
                 name=None, **kwargs):

        super(Transformer, self).__init__(name=name, **kwargs)
        
        assert embedding_size % num_attention_heads == 0

        self.attn = Attention(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            name='attention'
        )
        self.ln0 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='LayerNorm_mlp_ln0')
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='LayerNorm_mlp_ln1')
        self.intermediate_layer = get_dense(4 * embedding_size, name='intermediate')
        self.activation = tf.keras.layers.Activation(gelu)
        self.output_layer = get_dense(embedding_size, name='output')
        self.resid_drop = tf.keras.layers.Dropout(residual_dropout)

    def call(self, x, kv_cache=None, **kwargs):

        # https://github.com/karpathy/minGPT/blob/d100e2251a258ea6c72e59eeba83539567e8fc8c/mingpt/model.py#L96
        # https://github.com/openai/gpt-2/blob/0574c5708b094bfa0b0f6dfe3fd284d9a045acd9/src/model.py#L123

        attn, cached_kv = self.attn(self.ln0(x), kv_cache=kv_cache)
        x = x + attn

        z = self.ln1(x)
        z = self.intermediate_layer(z)
        z = self.activation(z)
        z = self.output_layer(z)
        z = self.resid_drop(z)

        x = x + z

        return x, cached_kv
    

class PositionEmbedding(tf.keras.layers.Layer):
    
    def __init__(self,
                 max_position_embeddings,
                 embedding_size, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)

        self.emb = self.add_weight(
            name="position_embeddings",
            dtype=tf.keras.backend.floatx(),
            shape=[max_position_embeddings, embedding_size])

    def call(self, x, kv_cache=None, **kwargs):
        seq_len = tf.shape(x)[1]
        if kv_cache is None:
            emb = tf.expand_dims(self.emb[:seq_len, :], 0)
        else:
            before = tf.shape(kv_cache)[-2]
            emb = tf.expand_dims(self.emb[before: before + seq_len, :], 0)
        return emb
    
    
class GPT(tf.keras.Model):

    def __init__(self,
                 vocab_size,
                 layer_size,
                 block_size,
                 embedding_dropout,
                 embedding_size,
                 num_attention_heads,
                 attention_dropout,
                 residual_dropout, **kwargs):

        super(GPT, self).__init__(**kwargs)

        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.pos_emb = PositionEmbedding(block_size, embedding_size)
        self.emb_drop = tf.keras.layers.Dropout(embedding_dropout)
        self.transformers = []
        for i in range(layer_size):
            self.transformers.append(Transformer(
                embedding_size=embedding_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                name=f'layer{i:02d}'
            ))
        self.final_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name='LayerNorm_final_norm')

    def call(self, x, kv_cache=None, use_cache=False, **kwargs):
        x = self.token_emb(x)
        x = self.emb_drop(x + self.pos_emb(x, kv_cache=kv_cache))

        cached_kvs = []
        for i, layer in enumerate(self.transformers):
            x, cached_kv = layer(
                x,
                kv_cache=kv_cache[i] if kv_cache is not None else None)
            cached_kvs.append(cached_kv)

        x = self.final_norm(x)
        emb_vec = tf.identity(self.token_emb.weights[0])
        x = tf.matmul(x, emb_vec,  transpose_b=True)
        if use_cache:
            return x, tf.stack(cached_kvs)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self.call(x)
            loss = tf.keras.backend.sparse_categorical_crossentropy(
                target=y, output=logits, from_logits=True
            )
            loss = tf.boolean_mask(loss, x > 0)
            # token average
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.optimizer._name == 'AdamW':
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables),
                decay_var_list=[
                    v
                    for v in self.trainable_variables
                    if 'bias' not in v.name.lower() and 'norm' not in v.name.lower()
                ]
            )
        else:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        ret = {
            m.name: m.result() for m in self.metrics
        }
        ret['loss'] = loss
        return ret
