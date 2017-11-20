#ifndef ESPRESSO_HPP
#define ESPRESSO_HPP

/**
    :----:
   C|====|
    |    |
    `----'  
 */
#include <vector>
#include <opencv2/core/cuda.hpp>
#include <caffe2/core/tensor.h>
#include <caffe2/core/context_gpu.h>

namespace espresso
{
using gpu_mat = cv::cuda::GpuMat;

// @brief extract 3 Channel BGR data
struct preprocess
{
    gpu_mat operator()(const gpu_mat arg,
                       cv::Size geometry) const;
};

// @brief exract a Float 3 Channel Matrix
struct split_single
{
    gpu_mat operator()(const gpu_mat arg) const;
};

struct split_vector
{
    std::vector<gpu_mat> operator()(const gpu_mat arg) const;
};

// @brief convert the Matrix to the correct type
struct convert
{
    gpu_mat operator()(const gpu_mat arg,
                       unsigned int channels) const;
};

// @brief make a float matrix from the BGR matrix
struct make_float
{
    gpu_mat operator()(const gpu_mat arg,
                       unsigned int channels) const;
};

// @brief copy byte-by-byte (device-to-device) for the tensor
struct copy_cuda_data
{
    // first overload uses a single GPU Matrix
    caffe2::TensorCUDA operator()(const gpu_mat arg,
                                  unsigned int channels,
                                  unsigned int rows,
                                  unsigned int cols) const;

    // second overload uses a vector of GPU matrices
    caffe2::TensorCUDA operator()(const std::vector<gpu_mat> arg,
                                  unsigned int channels,
                                  unsigned int rows,
                                  unsigned int cols) const;
};

// @brief create a CUDA tensor from split channel data
struct make_cuda_tensor
{
    // first overload uses a single GPU Matrix
    caffe2::TensorCUDA operator()(const gpu_mat arg,
                                  unsigned int channels,
                                  cv::Size geometry) const;
};

}
#endif
