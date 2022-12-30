"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        assert (a+self.scalar).dtype=='float32', "AddScalar: the result's dtype isn't float32"
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        assert (a*self.scalar).dtype=='float32', "MulScalar: the result's dtype isn't float32"
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        assert (a**self.scalar).dtype=='float32', "PowerScalar: the result's dtype isn't float32"
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        if self.scalar == 0:
            return init.zeros(lhs.shape)
        return out_grad * self.scalar * lhs ** (self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, -out_grad * lhs / (rhs * rhs))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        assert (a/self.scalar).dtype=='float32', "DivScalar: the result's dtype isn't float32"
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            if a.ndim < 2:
                raise Exception
            else:
                # return array_api.swapaxes(a,*range(a.ndim)[-2:])
                origin_axes = list(range(a.ndim))
                origin_axes[-2:] = origin_axes[-1:-3:-1]
                return a.permute(origin_axes).compact()
        else:
            origin_axes = list(range(a.ndim))
            assert len(self.axes)==2
            origin_axes[self.axes[0]], origin_axes[self.axes[1]] = origin_axes[self.axes[1]], origin_axes[self.axes[0]]
            return a.permute(origin_axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # always return a copy
        return array_api.reshape(a,self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad.reshape(lhs.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        temp = out_grad.sum(tuple(range(len(node.shape)-len(lhs.shape))))
        temp = out_grad.sum(tuple([i for i in range(len(lhs.shape)) if lhs.shape[i]==1]))
        return temp.reshape(lhs.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes,int): axes = tuple([axes])
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.summation(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        sshape = reduce_dimension(lhs.shape,self.axes)
        return out_grad.reshape(sshape).broadcast_to(lhs.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lres, rres = (out_grad @ rhs.transpose(),lhs.transpose() @ out_grad)
        ## in case of high dimension tensor, e.g. (6,6,5,4) @ (4,3)
        # but in fact, as the ndarray supports, len(axis) < 2
        if len(lres.shape) > len(lhs.shape):
            lres = lres.sum(tuple(range(len(lres.shape)-len(lhs.shape))))
        if len(rres.shape) > len(rhs.shape):
            rres = rres.sum(tuple(range(len(rres.shape)-len(rhs.shape))))
        return (lres,rres)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a > 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad * Tensor(lhs.realize_cached_data()>0,device=node.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Amax(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes,int): axes = tuple([axes])
        self.axes = axes

    def compute(self, a):
        return array_api.amax(a,self.axes)
    
    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        sshape = reduce_dimension(lhs.shape,self.axes)
        rs_out_grad = out_grad.reshape(sshape).broadcast_to(lhs.shape) 
        return rs_out_grad * Tensor(node.reshape(sshape).broadcast_to(lhs.shape).realize_cached_data() == lhs.realize_cached_data(),device=node.device)    

def amax(a, axes=None):
    return Amax(axes=axes)(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes,int): axes = tuple([axes])
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxZ = array_api.amax(Z,self.axes)
        return maxZ + array_api.log(array_api.summation(array_api.exp(Z - maxZ.reshape(reduce_dimension(Z.shape,self.axes)).broadcast_to(Z.shape)),self.axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        sshape = reduce_dimension(lhs.shape,self.axes)
        maxlhs = amax(lhs,self.axes).reshape(sshape).broadcast_to(lhs.shape)
        lhsrefine = exp(lhs - maxlhs)
        return out_grad.reshape(sshape).broadcast_to(lhs.shape) * (lhsrefine / summation(lhsrefine,self.axes).reshape(sshape).broadcast_to(lhs.shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad * 4 * ((exp(lhs) + exp(-lhs)) ** (-2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        inshape = args[0].shape
        nums = len(args)
        outshape = [nums] + list(inshape)
        pidx = [slice(None,None,None) for i in range(len(inshape))]
        ret = NDArray.make(outshape,device=args[0].device)
        for i,arg in enumerate(args):
            ret[tuple([i]+pidx)] = arg.compact()
        outidx = list(range(1,len(outshape)))
        outidx.insert(self.axis,0)
        return ret.permute(outidx).compact()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad,self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = A.shape
        nums = shape[self.axis]
        l = []
        for i in range(nums):
            idx = [slice(None,None,None) for i in range(len(shape))]
            idx[self.axis] = i
            temp = A[tuple(idx)].compact()
            newshape = [s for j,s in enumerate(temp.shape) if j!=self.axis]
            l.append(temp.reshape(newshape).compact())
        return tuple(l)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        newshape = list(a.shape)
        for axe in self.axes:
            newshape[axe] =newshape[axe] * (1+self.dilation)
        ret = NDArray.make(tuple(newshape),device=a.device)
        ret.fill(0)
        slices = [slice(None,None,1+self.dilation) if i in self.axes else slice(None,None,None) for i in range(ret.ndim)]
        ret[tuple(slices)] = a
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        newshape = list(a.shape)
        for axe in self.axes:
            assert newshape[axe] % (1+self.dilation) == 0
            newshape[axe] = newshape[axe] // (1+self.dilation)
        ret = NDArray.make(tuple(newshape),device=a.device)
        slices = [slice(None,None,1+self.dilation) if i in self.axes else slice(None,None,None) for i in range(ret.ndim)]
        ret = a[tuple(slices)]
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        assert A.ndim == B.ndim == 4
        A = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N,H,W,C_in = A.shape
        K_1,K_2,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
    
        inner_dim = K_1 * K_2 * C_in
        AA = NDArray.make(shape = (N, (H-K_1)//self.stride+1, (W-K_2)//self.stride+1, K_1, K_2, C_in),
                  strides = (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs),
                  device=A.device,
                  handle = A._handle).compact().reshape((N*((H-K_1)//self.stride+1)*((W-K_2)//self.stride+1),inner_dim)).compact()
        out = AA @ B.reshape((inner_dim, C_out)).compact()
        return out.reshape((N,(H-K_1)//self.stride+1,(W-K_2)//self.stride+1,C_out)).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        W_T = transpose(flip(W,(0,1)))
        K,_,_,C_out = W.shape
        # TODO: suppose H-K+2P % s == 0, otherwise padding needed after dilation
        assert (X.shape[1]-W.shape[0]+2*self.padding+1) % self.stride == 0
        assert (X.shape[2]-W.shape[1]+2*self.padding+1) % self.stride == 0
        out_grad_dilated = dilate(out_grad,(1,2),self.stride-1)
        Xres = conv(out_grad_dilated,W_T,stride=1,padding=K-self.padding-1)
        X_T = transpose(X,(0,3))
        Wres = conv(X_T,transpose(transpose(out_grad_dilated,(0,1)),(1,2)),stride=1,padding=self.padding)
        Wres = transpose(transpose(Wres,(0,1)),(1,2))
        return Xres,Wres
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)

## auxiliary function
def reduce_dimension(shape,axes):
    sshape = list(shape)
    for i in range(len(shape)):
        if axes is None or i in axes:
            sshape[i] = 1 
    return tuple(sshape)

