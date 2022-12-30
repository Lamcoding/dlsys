import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_nums = len(dataloader.dataset)

    if opt is None: 
        # eval mode
        model.eval()
        loss = 0
        acc = 0
        for i, (x,y) in enumerate(dataloader):
            out = model(x)
            loss += loss_fn(out,y).numpy()
            err += (out.numpy().argmax(axis=1) == y.numpy()).sum()
        loss = loss / (i+1)
        acc = acc / total_nums
        return acc.item(),loss.item()
    else:
        # train mode
        model.train()
        loss_np = 0
        acc_np = 0
        for i,(x,y) in enumerate(dataloader):
            print(f"batch {i}")
            opt.reset_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()
            loss_np += loss.numpy()
            acc_np += (out.numpy().argmax(axis=1) == y.numpy()).sum()
        loss_np = loss_np / (i+1)
        acc_np = acc_np / total_nums
        return acc_np.item(),loss_np.item()

    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss()):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    for epoch in range(n_epochs):
        print(f"start epoch {epoch}")
        acc,loss = epoch_general_cifar10(dataloader,model,loss_fn,opt)
        print(f"acc in epoch {epoch}: {acc}; loss is {loss}")
    return acc,loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    acc,loss = epoch_general_cifar10(dataloader,model,loss_fn)
    print(f"acc in the dataset is: {acc}; loss is {loss}")
    return acc,loss
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    kwargs = {"device":device,"dtype":dtype}
    nbatch,bs = data.shape
    state = None
    if opt: model.train()
    else: model.eval()
    total_loss = 0
    total_errors = 0
    total_batches = 0
    total_examples = 0
    for i in range(0,nbatch-1,seq_len):
        X,target = ndl.data.get_batch(data,i,seq_len,**kwargs)
        print(f"batchid: {i}====data shape: {data.shape}====target.shape: {target.shape}")
        if opt:
            opt.reset_grad()
            logits,state = model(X,state)
            # print(f"logits device: {logits.device}")
            loss = loss_fn(logits,target)
            loss.backward()
            if clip: opt.clip_grad_norm(max_norm=clip)
            opt.step()
            if isinstance(state,tuple):
                state = tuple([s.detach() for s in state])
            else: state = state.detach()
        else:
            logits,state = model(X,state)
            loss = loss_fn(logits,target)
        logits_np = logits.numpy()
        y_pred = np.argmax(logits_np,axis=1)
        errors = np.not_equal(y_pred,target).sum()

        total_loss += loss.numpy()
        total_errors += errors
        total_batches += 1
        total_examples += target.shape[0]
    
    avg_loss = total_loss / total_batches
    avg_acc = 1 - total_errors / total_examples
    return avg_acc,avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn = loss_fn()
    for epoch in range(n_epochs):
        acc,loss = epoch_general_ptb(data,model,seq_len,loss_fn,opt,clip,device,dtype)
        print(f"Train epoch: {epoch}======acc: {acc}====loss: {loss}")
    return acc,loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = loss_fn()
    acc, loss = epoch_general_ptb(data,model,seq_len,loss_fn,device=device,dtype=dtype)
    print("acc: {acc}======loss: {loss}")
    return acc,loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
