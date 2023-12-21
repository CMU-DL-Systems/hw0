#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


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
    size_t index = 0;
    while(index < m){
        const float *X_batch = X + index*n;
        const unsigned char *y_batch = y + index;
        size_t m_batch = (m - index >= batch) ? batch : m - index;
        index += batch;

        // dot, exp and sum
        float *Z = new float[m_batch*k];
        float sums[m_batch];
        for(size_t i = 0; i < m_batch; i++){
            float sum = 0;
            for(size_t j = 0; j < k; j++){
                float res = 0;
                for(size_t l = 0; l < n; l++){
                    res += X_batch[i*n + l] * theta[l*k + j];
                }
                Z[i*k + j] = exp(res);
                sum += Z[i*k + j];
            }
            sums[i] = sum;
        }

        // normalization and substract Iy
        short *Iy = new short[m_batch*k];
        for(size_t i = 0; i < m_batch; i++){
            for(size_t j = 0; j < k; j++){
                Iy[i*k + j] = y_batch[i] == j ? 1 : 0;

                Z[i*k + j] /= sums[i];
                Z[i*k + j] -= Iy[i*k + j];
            }
        }

        // Update theta: dot X_batch.T and Z (Z - Iy), subtract lr*gradient 
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < k; j++){
                float res = 0;
                for(size_t l = 0; l < m_batch; l++){
                    res += X_batch[l*n + i] * Z[l*k + j];
                }
                theta[i*k + j] -= lr * res / m_batch;
            }
        }
    }
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
