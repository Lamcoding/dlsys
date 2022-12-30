"""The module.
"""
from typing import List, Callable, Any
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
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
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
        batchsl = ops.logsumexp(logits,axes=(1,)) - (init.one_hot(logits.shape[1],y) * logits).sum((1,))
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
    def __init__(self, p = 0.5):
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



