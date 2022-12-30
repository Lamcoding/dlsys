#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

void matrixMulti(const float* a, const float* b, float* c,size_t m, size_t n,size_t k,int abegin){
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            float result = 0.0;
            for(int temp=0;temp<n;temp++){
                result += a[(i+abegin)*n+temp] * b[temp*k+j];
            }
            c[i*k+j] = result;
        }
    }
}

void transpose(const float* a,float* b,size_t m,size_t n,int abegin){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            b[j*m+i] = a[(i+abegin)*n+j];
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t batchloop = m / batch + (m%batch!=0);
    float * z = new float[batch*k];
    float * temp = new float[n*batch];
    float * grad = new float[n*k];
    for(size_t batchid=0;batchid<batchloop;batchid++){
        size_t batch_size = ((batchid+1)*batch)<m?batch:m-batchid*batch;
        matrixMulti(X,theta,z,batch_size,n,k,batchid*batch);
        
        for(int i=0;i<batch_size;i++){
            double totalsum = 0.0;
            for(int j=0;j<k;j++){
                z[i*k+j] = exp(z[i*k+j]);
                totalsum += z[i*k+j];
            }
            for(int j=0;j<k;j++)
                z[i*k+j] /= totalsum;
            z[i*k+y[batchid*batch+i]] -= 1;
        }
        transpose(X,temp,batch_size,n,batchid*batch);
        matrixMulti(temp,z,grad,n,batch_size,k,0);

        for(int i=0;i<n;i++){
            for(int j=0;j<k;j++){
                theta[i*k+j] -= lr/batch_size * grad[i*k+j];
            }
        }
    }
    delete(z);
    delete(temp);
    delete(grad);

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
