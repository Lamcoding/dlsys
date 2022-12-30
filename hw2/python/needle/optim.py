"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for k,param in enumerate(self.params):
            if param.grad is None:
                continue
            d_p = param.grad.data + param.data * self.weight_decay
            if k not in self.u:
                self.u[k] = ndl.init.zeros(*param.shape)
            self.u[k].data = self.momentum * self.u[k] + (1. - self.momentum) * d_p
            
            param.data = param.data -  self.lr * self.u[k].data
        # if self.momentum != 0.:
        #     print(self.u)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t = self.t + 1
        for k,param in enumerate(self.params):
            if param.grad is None:
                continue
            d_p = param.grad + self.weight_decay * param
            if k not in self.m:
                self.m[k] = ndl.init.zeros(*param.shape)
            self.m[k].data = self.m[k] * self.beta1 + (1. - self.beta1) * d_p
            if k not in self.v:
                self.v[k] = ndl.init.zeros(*param.shape)
            self.v[k].data = self.v[k] * self.beta2 + (1. - self.beta2) * d_p ** 2
            mk_cor = self.m[k] / (1. - self.beta1 ** self.t)
            vk_cor = self.v[k] / (1. - self.beta2 ** self.t)
            param.data = param - self.lr * (mk_cor / (vk_cor ** 0.5 + self.eps))

        ### END YOUR SOLUTION
