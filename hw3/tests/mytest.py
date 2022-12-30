from needle import backend_ndarray as nd

x = nd.NDArray([1,2,3],device=nd.cuda())
y = nd.NDArray([1,2,1],device=nd.cuda())

print(x-y)
