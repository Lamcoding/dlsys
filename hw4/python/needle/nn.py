"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features,out_features,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features,1,device=device,dtype=dtype,requires_grad=True).reshape((1,out_features)).data) if bias else None
        # print("self.bias ndim",self.bias.shape if self.bias is not None else "None")
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ret_nobias = X @ self.weight
        return ret_nobias if self.bias is None else ret_nobias + self.bias.broadcast_to(ret_nobias.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        assert len(X.shape) >= 2, "the input to Flatten mush in batch form"
        batch = X.shape[0]
        other = X.shape[1:]
        size = 1
        for i in other:
            size *= i
        return X.reshape((batch,size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one = init.ones_like(x)
        return one / (one + ops.exp(-x))
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ret = x
        for module in self.modules:
            ret = module(ret)
        return ret
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        ## assume logits is 2D (num_examples,feature_nums)
        ## assume y is a 1D tensor
        batchsl = ops.logsumexp(logits,axes=(1,)) - (init.one_hot(logits.shape[1],y,device=logits.device,dtype=logits.dtype) * logits).sum((1,))
        return batchsl.sum() / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True))
        self.running_mean = init.zeros(dim,device=device,dtype=dtype)
        self.running_var = init.ones(dim,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        sshape = ops.reduce_dimension(x.shape,(0,))
        if self.training:
            mean_observed = x.sum((0,)) / x.shape[0]
            var_observed = ((x-mean_observed.reshape(sshape).broadcast_to(x.shape)) ** 2).sum((0,)) / x.shape[0]
            self.running_mean.data = (1-self.momentum) * self.running_mean + self.momentum * mean_observed
            self.running_var.data = (1-self.momentum) * self.running_var + self.momentum * var_observed
            unbiased_x = (x - mean_observed.reshape(sshape).broadcast_to(x.shape)) / (var_observed.reshape(sshape).broadcast_to(x.shape) + self.eps) ** 0.5
        else:
            unbiased_x = (x - self.running_mean.reshape(sshape).broadcast_to(x.shape)) / (self.running_var.reshape(sshape).broadcast_to(x.shape) + self.eps) ** 0.5
        return self.weight.reshape(sshape).broadcast_to(x.shape) * unbiased_x + self.bias.reshape(sshape).broadcast_to(x.shape)
        # return unbiased_x
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert x.shape[1] == self.dim, "the feature vector's dim is not equal to self.dim"
        sshape = ops.reduce_dimension(x.shape,(1,))
        def cal_mean(a):
            return (a.sum((1,)) / self.dim).reshape(sshape).broadcast_to(a.shape)
        mean = cal_mean(x)
        unbiased_x = x - mean
        variance = cal_mean(unbiased_x ** 2)
        wshape = ops.reduce_dimension(x.shape,(0,))
        return self.weight.reshape(wshape).broadcast_to(x.shape) * (unbiased_x / ((variance + self.eps) ** 0.5)) + self.bias.reshape(wshape).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training and self.p < 1.:
          return x * init.randb(*x.shape,p=1. - self.p,dtype="float32") / (1. - self.p)
        elif self.training and self.p == 1.:
          return x * init.zeros(*x.shape)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        reception_field_size = kernel_size * kernel_size
        self.weight = Parameter(init.kaiming_uniform(in_channels*reception_field_size,out_channels*reception_field_size,(kernel_size,kernel_size,in_channels,out_channels),device=device,dtype=dtype,requires_grad=True))
        interval = 1.0/(in_channels * kernel_size**2)**0.5
        if bias:
            self.bias = Parameter(init.rand(out_channels,low=-interval,high=interval,device=device,dtype=dtype,requires_grad=True))
        else: self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Ensure nn.Conv works for (N, C, H, W) tensors even though we implemented the conv op for (N, H, W, C) tensors
        X = x.transpose((1,2)).transpose((2,3))
        assert X.shape[3] == self.in_channels
        # TODO and BUG: solve (H-K+2p+1)//s +1 = H how to understand k//2
        # as the PyTorch Conv2d, same padding only supports stride=1
        ret = ops.conv(X,self.weight,stride=self.stride,padding=self.kernel_size//2)
        if self.bias:
            ret = ret + self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(ret.shape)
        return ret.transpose((2,3)).transpose((1,2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.bias = bias
        assert nonlinearity == 'tanh' or nonlinearity == 'relu', "Nonlinearity only supports tanh or relu"
        kwargs = {"device":device,"dtype":dtype}
        self.kwargs = kwargs
        k = 1.0 / hidden_size 
        self.bias_ih = Parameter(init.rand(hidden_size,low=-k**0.5,high=k**0.5,requires_grad=True,**kwargs)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size,low=-k**0.5,high=k**0.5,requires_grad=True,**kwargs)) if bias else None
        self.W_ih = Parameter(init.rand(input_size,hidden_size,low=-k**0.5,high=k**0.5,requires_grad=True,**kwargs))
        self.W_hh = Parameter(init.rand(hidden_size,hidden_size,low=-k**0.5,high=k**0.5,requires_grad=True,**kwargs))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs,input_size = X.shape
        assert X.shape[1] == self.input_size
        out = X @ self.W_ih
        if self.bias:
            out = out + self.bias_ih.reshape((1,self.hidden_size)).broadcast_to((bs,self.hidden_size)) + self.bias_hh.reshape((1,self.hidden_size)).broadcast_to((bs,self.hidden_size))
        if h is None:
            h = init.zeros(bs,self.hidden_size,**self.kwargs)
        assert h.shape[1] == self.hidden_size
        assert h.shape[0] == X.shape[0]
        out = out + h @ self.W_hh
        return ops.tanh(out) if self.nonlinearity == 'tanh' else ops.relu(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [RNNCell(input_size,hidden_size,bias,nonlinearity,device,dtype) if i==0 else RNNCell(hidden_size,hidden_size,bias,nonlinearity,device,dtype) for i in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len,bs,input_size =X.shape
        if h0:
            h0s = [x for x in ops.split(h0,0)]
        else:
            h0s = [None for i in range(self.num_layers)]
        seq_ins = ops.split(X,0)
        seq_outs = []
        for seq_in in seq_ins:
            for i in range(self.num_layers):
                seq_in = self.rnn_cells[i](seq_in,h0s[i])
                h0s[i] = seq_in
            seq_outs.append(seq_in)
        return ops.stack(seq_outs,0), ops.stack(h0s,0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        kwargs = {"device":device,"dtype":dtype}
        self.kwargs = kwargs
        bound = (1.0 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size,4*hidden_size,low=-bound,high=bound,requires_grad=True,**kwargs))
        self.W_hh = Parameter(init.rand(hidden_size,4*hidden_size,low=-bound,high=bound,requires_grad=True,**kwargs))
        if self.bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size,low=-bound,high=bound,requires_grad=True,**kwargs)) 
            self.bias_hh = Parameter(init.rand(4*hidden_size,low=-bound,high=bound,requires_grad=True,**kwargs))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch,input_size = X.shape
        out = X @ self.W_ih
        if self.bias:
            out = out + self.bias_ih.reshape((1,4*self.hidden_size)).broadcast_to((batch,4*self.hidden_size))
            out = out + self.bias_hh.reshape((1,4*self.hidden_size)).broadcast_to((batch,4*self.hidden_size))
        h0 = init.zeros(batch,self.hidden_size,**self.kwargs)
        c0 = init.zeros(batch,self.hidden_size,**self.kwargs)
        if h:
            h0, c0 = h
            if h0 is None: h0 = init.zeros(batch,self.hidden_size,**self.kwargs)
            if c0 is None: c0 = init.zeros(batch,self.hidden_size,**self.kwargs)
        out = out + h0 @ self.W_hh
        hs = [x for x in ops.split(out,1)]
        i,f,g,o = [ops.stack(hs[i:i+self.hidden_size],1) for i in range(0,len(hs),self.hidden_size)]
        sg = Sigmoid()
        i = sg(i)
        f = sg(f)
        g = ops.tanh(g)
        o = sg(o)
        c_out = f * c0 + i * g 
        h_out = o * ops.tanh(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        kwargs = {"device":device,"dtype":dtype}
        self.lstm_cells = [LSTMCell(hidden_size,hidden_size,bias,**kwargs) for i in range(1,num_layers)]
        self.lstm_cells.insert(0,LSTMCell(input_size,hidden_size,bias,**kwargs))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_ins = ops.split(X,0)
        outputs = []
        h0s = [None for x in range(self.num_layers)]
        c0s = [None for x in range(self.num_layers)]
        if h:
            h0,c0 = h
            if h0: h0s = [x for x in ops.split(h0,0)] 
            if c0: c0s = [x for x in ops.split(c0,0)]
        for seq_in in seq_ins:
            for k in range(self.num_layers):
                h0s[k],c0s[k] = self.lstm_cells[k](seq_in,(h0s[k],c0s[k]))
                seq_in = h0s[k]
            outputs.append(h0s[k])
        return ops.stack(outputs,0), (ops.stack(h0s,0), ops.stack(c0s,0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kwargs = {"device":device,"dtype":dtype}
        self.weight = Parameter(init.randn(num_embeddings,embedding_dim,requires_grad=True,**self.kwargs))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len,bs = x.shape
        y = x.reshape((seq_len*bs,))
        onehot_x = init.one_hot(self.num_embeddings,y,**self.kwargs)
        ret = onehot_x @ self.weight
        return ret.reshape((seq_len,bs,self.embedding_dim))
        ### END YOUR SOLUTION
