import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
              nn.Linear(in_features=dim,out_features=hidden_dim),
              norm(hidden_dim),
              nn.ReLU(),
              nn.Dropout(drop_prob),
              nn.Linear(in_features=hidden_dim,out_features=dim),
              norm(dim)
            )
    return nn.Sequential(nn.Residual(fn),nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
              nn.Linear(in_features=dim,out_features=hidden_dim),
              nn.ReLU(),
              *[ResidualBlock(dim=hidden_dim,hidden_dim=hidden_dim//2,norm=norm,drop_prob=drop_prob) for i in range(num_blocks)],
              nn.Linear(in_features=hidden_dim,out_features=num_classes)
            )
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    ## specify the MNIST dataset
    loss_fn = nn.SoftmaxLoss()
    total_nums = len(dataloader.dataset)

    if opt is None: 
        # eval mode
        model.eval()
        loss = 0
        err = 0
        for i, (x,y) in enumerate(dataloader):
            out = model(x)
            loss += loss_fn(out,y).numpy()
            err += (out.numpy().argmax(axis=1) != y.numpy()).sum()
        loss = loss / (i+1)
        err = err / total_nums
        return err.item(),loss.item()
    else:
        # train mode
        model.train()
        loss_np = 0
        err_np = 0
        for i,(x,y) in enumerate(dataloader):
            opt.reset_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()
            loss_np += loss.numpy()
            err_np += (out.numpy().argmax(axis=1) != y.numpy()).sum()
        loss_np = loss_np / (i+1)
        err_np = err_np / total_nums
        return err_np.item(),loss_np.item()


    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(dim=28*28,hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    mnist_train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir,"train-images-idx3-ubyte.gz"),
                           os.path.join(data_dir,"train-labels-idx1-ubyte.gz"))
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                            batch_size=batch_size,
                            shuffle=True)
    mnist_test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir,"t10k-images-idx3-ubyte.gz"),
                           os.path.join(data_dir,"t10k-labels-idx1-ubyte.gz"))
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    for i in range(epochs):
        train_err,train_loss = epoch(mnist_train_dataloader,model,opt=opt)
    
    test_err, test_loss = epoch(mnist_test_dataloader,model)
    return train_err,train_loss,test_err,test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
