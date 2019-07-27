#! -*- coding: utf-8 -*-

from keras.layers import *
import keras.backend as K


def to_mask(x, mask, mode='mul'):
    """通用mask函数
    """
    if mask is None:
        return x
    else:
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10


def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = K.int_shape(x)[-1]
    seq_len = K.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = K.temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = K.concatenate(xs, 2)
    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs


class Attention(OurLayer):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def call(self, inputs):
        q, k, v = inputs[: 3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = to_mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right is True:
                ones = K.ones_like(a[: 1, : 1])
                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
                a = a - mask
            else:
                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]
                mask = (1 - K.constant(self.mask_right)) * 1e10
                mask = K.expand_dims(K.expand_dims(mask, 0), 0)
                self.mask = mask
                a = a - mask
        a = K.softmax(a)
        self.a = a
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = to_mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class SelfAttention(OurLayer):
    """多头自注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            self.mask_right
        )
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
            o = self.reuse(self.attention, [x, x, x, x_mask, x_mask])
        else:
            x = inputs
            o = self.reuse(self.attention, [x, x, x])
        return o
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class AtrousSelfAttention(OurLayer):
    """空洞多头自注意力机制
    说明：每个元素只跟相对距离为rate的倍数的元素有关联。
    """
    def __init__(self, heads, size_per_head, rate=1,
                 key_size=None, mask_right=False, **kwargs):
        super(AtrousSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.rate = rate
        self.mask_right = mask_right
    def build(self, input_shape):
        super(AtrousSelfAttention, self).build(input_shape)
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            self.mask_right
        )
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        # 变换shape
        x = K.reshape(x, (-1, new_seq_len // self.rate, self.rate, seq_dim))
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        x = K.reshape(x, (-1, new_seq_len // self.rate, seq_dim))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1))
            x_mask = K.permute_dimensions(x_mask, (0, 2, 1, 3))
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, 1))
        # 做attention
        if x_mask is not None:
            x = self.reuse(self.attention, [x, x, x, x_mask, x_mask])
        else:
            x = self.reuse(self.attention, [x, x, x])
        # 恢复shape
        x = K.reshape(x, (-1, self.rate, new_seq_len // self.rate, self.out_dim))
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        x = K.reshape(x, (-1, new_seq_len, self.out_dim))
        x = x[:, : - pad_len]
        return x
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class LocalSelfAttention(OurLayer):
    """局部多头自注意力机制
    说明：每个元素只跟相对距离不超过neighbors的元素有关联，这里的rate
    是真正的膨胀率（跟膨胀卷积一样），如果不了解可以忽略，默认为1就好。
    """
    def __init__(self, heads, size_per_head, neighbors=1, rate=1,
                 key_size=None, mask_right=False, **kwargs):
        super(LocalSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.neighbors = neighbors
        self.rate = rate
        self.mask_right = mask_right
    def build(self, input_shape):
        super(LocalSelfAttention, self).build(input_shape)
        if self.mask_right:
            mask_right = np.ones((1, 1 + 2 * self.neighbors))
            mask_right[:, - self.neighbors : ] = 0
        else:
            mask_right = self.mask_right
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            mask_right
        )
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        xp = extract_seq_patches(x, kernel_size, self.rate)
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 变换shape
        seq_len = K.shape(x)[1]
        seq_dim = K.int_shape(x)[-1]
        x = K.reshape(x, (-1, 1, seq_dim))
        xp = K.reshape(xp, (-1, kernel_size, seq_dim))
        if x_mask is not None:
            xp_mask = K.reshape(xp_mask, (-1, kernel_size, 1))
        # 做attention
        if x_mask is not None:
            x = self.reuse(self.attention, [x, xp, xp, xp_mask])
        else:
            x = self.reuse(self.attention, [x, xp, xp])
        # 恢复shape
        x = K.reshape(x, (-1, seq_len, self.out_dim))
        x = to_mask(x, x_mask, 'mul')
        return x
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class SparseSelfAttention(OurLayer):
    """稀疏多头自注意力机制
    来自文章《Generating Long Sequences with Sparse Transformers》
    说明：每个元素只跟相对距离为rate的倍数的元素、以及相对距离不超过rate的元素有关联。
    """
    def __init__(self, heads, size_per_head, rate=2,
                 key_size=None, mask_right=False, **kwargs):
        super(SparseSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        assert rate != 1, u'if rate=1, please use SelfAttention directly'
        self.rate = rate
        self.neighbors = rate - 1
        self.mask_right = mask_right
    def build(self, input_shape):
        super(SparseSelfAttention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        x = K.reshape(x, (-1, new_seq_len, seq_dim)) # 经过padding后shape可能变为None，所以重新声明一下shape
        # 线性变换
        qw = self.reuse(self.q_dense, x)
        kw = self.reuse(self.k_dense, x)
        vw = self.reuse(self.v_dense, x)
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        kwp = extract_seq_patches(kw, kernel_size, self.rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, self.rate) # shape=[None, seq_len, kernel_size, out_dim]
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 形状变换
        qw = K.reshape(qw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        kw = K.reshape(kw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        vw = K.reshape(vw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.size_per_head))
        kwp = K.reshape(kwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.key_size))
        vwp = K.reshape(vwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.size_per_head))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1, 1))
            xp_mask = K.reshape(xp_mask, (-1, new_seq_len // self.rate, self.rate, kernel_size, 1, 1))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 3, 2, 1, 4)) # shape=[None, heads, r, seq_len // r, size]
        kw = K.permute_dimensions(kw, (0, 3, 2, 1, 4))
        vw = K.permute_dimensions(vw, (0, 3, 2, 1, 4))
        qwp = K.expand_dims(qw, 4)
        kwp = K.permute_dimensions(kwp, (0, 4, 2, 1, 3, 5)) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = K.permute_dimensions(vwp, (0, 4, 2, 1, 3, 5))
        if x_mask is not None:
            x_mask = K.permute_dimensions(x_mask, (0, 3, 2, 1, 4))
            xp_mask = K.permute_dimensions(xp_mask, (0, 4, 2, 1, 3, 5))
        # Attention1
        a = K.batch_dot(qw, kw, [4, 4]) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        a = to_mask(a, x_mask, 'add')
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        if self.mask_right:
            ones = K.ones_like(a[: 1, : 1, : 1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        # Attention2
        ap = K.batch_dot(qwp, kwp, [5, 5]) / self.key_size**0.5
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if x_mask is not None:
            ap = to_mask(ap, xp_mask, 'add')
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if self.mask_right:
            mask = np.ones((1, kernel_size))
            mask[:, - self.neighbors : ] = 0
            mask = (1 - K.constant(mask)) * 1e10
            for _ in range(4):
                mask = K.expand_dims(mask, 0)
            ap = ap - mask
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = K.concatenate([a, ap], -1)
        A = K.softmax(A)
        a, ap = A[..., : K.shape(a)[-1]], A[..., K.shape(a)[-1] : ]
        # 完成输出1
        o1 = K.batch_dot(a, vw, [4, 3])
        # 完成输出2
        ap = K.expand_dims(ap, -2)
        o2 = K.batch_dot(ap, vwp, [5, 4])
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        o = to_mask(o, x_mask, 'mul')
        o = K.permute_dimensions(o, (0, 3, 2, 1, 4))
        o = K.reshape(o, (-1, new_seq_len, self.out_dim))
        o = o[:, : - pad_len]
        return o
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class TrainablePositionEmbedding(OurLayer):
    """定义位置Embedding，直接训练出来
    """
    def __init__(self, maxlen, v_dim,
                 merge_mode='add', **kwargs):
        super(TrainablePositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.v_dim = v_dim
        self.merge_mode = merge_mode
    def build(self, input_shape):
        super(TrainablePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.maxlen, self.v_dim),
            initializer='zeros'
        )
    def call(self, inputs):
        """允许传入r（当前位置id）来得到相对位置向量
        """
        if isinstance(inputs, list):
            x, r = inputs
        else:
            x, r = inputs, 0
        pid = K.arange(K.shape(x)[1])
        pid = K.expand_dims(pid, 0)
        pid = K.tile(pid, [K.shape(x)[0], 1])
        pid = K.abs(pid - K.cast(r, 'int32'))
        pv = K.gather(self.embeddings, pid)
        if self.merge_mode == 'add':
            return pv + x
        else:
            return K.concatenate([x, pv])
    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return (input_shape[0], input_shape[1], input_shape[2] + self.v_dim)


class SinCosPositionEmbedding(Layer):
    """Google提出来的Sin-Cos形式的位置Embedding
    """
    def __init__(self, v_dim,
                 merge_mode='add', **kwargs):
        super(SinCosPositionEmbedding, self).__init__(**kwargs)
        self.v_dim = v_dim
        self.merge_mode = merge_mode
    def call(self, inputs):
        """允许传入r（当前位置id）来得到相对位置向量
        """
        if isinstance(inputs, list):
            x, r = inputs
        else:
            x, r = inputs, 0
        pid = K.arange(K.shape(x)[1])
        pid = K.expand_dims(pid, 0)
        pid = K.tile(pid, [K.shape(x)[0], 1])
        pid = K.abs(pid - K.cast(r, 'int32'))
        pv = self.idx2pos(pid)
        if self.merge_mode == 'add':
            return pv + x
        else:
            return K.concatenate([x, pv])
    def idx2pos(self, pid):
        pid = K.cast(pid, 'float32')
        pid = K.expand_dims(pid, 2)
        pj = 1. / K.pow(10000., 2. / self.v_dim * K.arange(self.v_dim // 2, dtype='float32'))
        pj = K.expand_dims(pj, 0)
        pv = K.dot(pid, pj)
        pv1, pv2 = K.sin(pv), K.cos(pv)
        pv1, pv2 = K.expand_dims(pv1, 3), K.expand_dims(pv2, 3)
        pv = K.concatenate([pv1, pv2], 3)
        return K.reshape(pv, (K.shape(pv)[0], K.shape(pv)[1], self.v_dim))
    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return (input_shape[0], input_shape[1], input_shape[2] + self.v_dim)
