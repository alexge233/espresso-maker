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
                       cv::Size geometry = cv::Size(224,224)) const;
};

/**
 * @brief splits the BRG float data from @param arg
 *        into the tensor
 */
struct channel_split
{
    void operator()(caffe2::TensorCUDA & tensor,
                    const gpu_mat arg) const;
};

// @brief convert the Matrix to the correct type
struct convert
{
    gpu_mat operator()(const gpu_mat arg,
                       unsigned int channels = 3) const;
};

// @brief make a float matrix from the BGR matrix
struct make_float
{
    gpu_mat operator()(const gpu_mat arg,
                       unsigned int channels = 3) const;
};

// @brief create a CUDA tensor from split channel data
struct make_cuda_tensor
{
    // first overload uses a single GPU Matrix
    void operator()(caffe2::TensorCUDA & tensor,
                    const gpu_mat arg,
                    unsigned int channels = 3,
                    cv::Size geometry = cv::Size(224, 224)) const;
};

}
#endif
