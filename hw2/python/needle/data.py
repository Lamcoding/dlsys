import numpy as np
import gzip
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    ################################################################
    # first 4 bytes is a magic number
    image_f = gzip.open(image_filesname,'r')
    magic_number = int.from_bytes(image_f.read(4), 'big')
    # second 4 bytes is the number of images
    image_count = int.from_bytes(image_f.read(4), 'big')
    # third 4 bytes is the row count
    row_count = int.from_bytes(image_f.read(4), 'big')
    # fourth 4 bytes is the column count
    column_count = int.from_bytes(image_f.read(4), 'big')
    # rest is the image pixel data, each pixel is stored as an unsigned byte
    # pixel values are 0 to 255
    image_data = image_f.read()
    images = np.frombuffer(image_data, dtype=np.uint8)\
        .reshape((image_count, row_count*column_count))\
        .astype(np.float32)
    images = images / 255.0
    #################################################################
    label_f = gzip.open(label_filename, 'r')
    magic_number = int.from_bytes(label_f.read(4), 'big')
    # second 4 bytes is the number of labels
    label_count = int.from_bytes(label_f.read(4), 'big')
    # rest is the label data, each label is stored as unsigned byte
    # label values are 0 to 9
    label_data = label_f.read()
    labels = np.frombuffer(label_data, dtype=np.uint8)
    #################################################################
    # print(images.shape,labels.shape)
    return (images, labels)
    ### END YOUR CODE

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        img_copy = img[:]
        if flip_img:
            img_copy[:,:,:] = img_copy[:,::-1,:]
        return img_copy
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        shape = img.shape
        ret = img.copy()
        # if shift_x > 0:
        #     ret[0:shift_x,:,:] = 0
        #     ret[shift_x:,:,:] = img[shift_x:,:]
        # else:
        #     ret[0:shift_x,:,:] = img[-shift_x:,:,:]
        #     ret[shift_x:,:,:] = 0
        shift_x = -shift_x
        shift_y = -shift_y
        # print(shift_x,shift_y)
        # print("img channel",img.transpose(2,0,1)[2])
        if shift_x != 0:
          ret[0:shift_x,:,:] = 0 if shift_x > 0 else img[-shift_x:,:,:]
          ret[shift_x:,:,:] = img[0:-shift_x,:,:] if shift_x > 0 else 0
        rett = ret.copy()
        if shift_y != 0:
          ret[:,0:shift_y,:] = 0 if shift_y > 0 else rett[:,-shift_y:,:]
          ret[:,shift_y:,:] = rett[:,0:-shift_y,:] if shift_y > 0 else 0
        # print("ret channel",ret.transpose(2,0,1)[2])
        return ret
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        else:
            arr = np.arange(len(dataset))
            np.random.shuffle(arr)
            self.ordering = np.array_split(arr,
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.index = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index == len(self.ordering):
            raise StopIteration

        self.index = self.index + 1
        return [Tensor(x) for x in self.dataset[self.ordering[self.index-1]]]
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms

        self.images, self.labels = parse_mnist(self.image_filename,self.label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.images[index]
        imgs = imgs.reshape((-1,28,28,1))
        lbs = self.labels[index]
        if self.transforms is not None:
            for t in self.transforms:
                for i,img in enumerate(imgs):
                    imgs[i] = t(img)
        if imgs.shape[0] == 1:
            imgs = imgs[0]
        imgs = imgs.reshape((-1,784))
        return imgs,lbs
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
