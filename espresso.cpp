#include "espresso.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <Eigen/Eigen>
#include <caffe2/core/init.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/context_gpu.h>

namespace espresso
{
gpu_mat preprocess::operator()(const gpu_mat arg,
                               cv::Size geometry) const
{
    cv::cuda::GpuMat resized;
    if (arg.size() != geometry) {
        cv::cuda::resize(arg, resized, geometry);
    }
    else {
        resized = arg;
    }
    cv::Rect crop((resized.cols - geometry.height) / 2,
                  (resized.rows - geometry.width) / 2, 
                   geometry.height,
                   geometry.width);
    return resized;
}

gpu_mat convert::operator()(const gpu_mat arg,
                            unsigned int channels) const
{
    cv::cuda::GpuMat sample;
    if (arg.channels() == 3 && channels == 1) {
        cv::cuda::cvtColor(arg, sample, cv::COLOR_BGR2GRAY);
    }
    else if (arg.channels() == 4 && channels == 1) {
        cv::cuda::cvtColor(arg, sample, cv::COLOR_BGRA2GRAY);
    }
    else if (arg.channels() == 4 && channels == 3) {
        cv::cuda::cvtColor(arg, sample, cv::COLOR_BGRA2BGR);
    }
    else if (arg.channels() == 1 && channels == 3) {
        cv::cuda::cvtColor(arg, sample, cv::COLOR_GRAY2BGR);
    }
    else {
        return arg;
    }
    return sample;
}

gpu_mat make_float::operator()(const gpu_mat arg,
                               unsigned int channels) const
{
    cv::cuda::GpuMat mfloat;
    if (channels == 3) {
        arg.convertTo(mfloat, CV_32FC3, 1.0, -128);
    }
    else {
        arg.convertTo(mfloat, CV_32FC1, 1.0, -128);
    }
    return mfloat;
}

gpu_mat split_single::operator()(const gpu_mat arg) const
{
    cv::cuda::GpuMat result(3);
    cv::cuda::split(arg, &result);
    return result;
}

std::vector<gpu_mat> split_vector::operator()(const gpu_mat arg) const
{
    std::vector<cv::cuda::GpuMat> input_channels(3);
    cv::cuda::split(arg, input_channels);
    return input_channels;
}

caffe2::TensorCUDA copy_cuda_data::operator()(const std::vector<gpu_mat> arg,
                                              unsigned int channels,
                                              unsigned int rows,
                                              unsigned int cols) const
{
    caffe2::TensorCUDA tensor;
    std::vector<caffe2::TIndex> dims({1, channels, rows, cols});
    tensor.Resize(dims);
    
    // the tensor is (224 * 224) = 50176 * 3 (channels) = 150528
    // or simply put (rows * cols) * channels
    float * t_ptr = tensor.mutable_data<float>();

    // iterate arg and copy one at a time
    unsigned int i = 0;
    for (const auto & gpu_mat : arg) {
        // pointer to GPU Matrix
        const void * src = gpu_mat.ptr<float>();
        // size of GPU Matrix = rows * cols
        size_t count = (gpu_mat.rows * gpu_mat.cols);
        // set tensor pointer
        void * dst = &t_ptr[i];
        // copy from device to device
        auto res = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
        if (res != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorName(res));
        }
        i += count;
    }
    return tensor;
}

caffe2::TensorCUDA copy_cuda_data::operator()(const gpu_mat arg,
                                              unsigned int channels,
                                              unsigned int rows,
                                              unsigned int cols) const
{
    caffe2::TensorCUDA tensor;
    std::vector<caffe2::TIndex> dims({1, channels, rows, cols});
    tensor.Resize(dims);
    
    // the tensor is (224 * 224) = 50176 * 3 (channels) = 150528
    // or simply put (rows * cols) * channels
    float * t_ptr = tensor.mutable_data<float>();
    // pointer to GPU Matrix
    const void * src = arg.ptr<float>();
    // size of GPU Matrix = rows * cols
    size_t count = (arg.rows * arg.cols);
    // set tensor pointer
    void * dst = &t_ptr;
    // copy from device to device
    auto res = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    if (res != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorName(res));
    }
    return tensor;
}

caffe2::TensorCUDA make_cuda_tensor::operator()(const gpu_mat image,
                                                unsigned int channels,
                                                cv::Size geometry) const
{
    cv::cuda::GpuMat resized;
    if (image.size() != geometry)
        cv::cuda::resize(image, resized, geometry);
    else
        resized = image;

    cv::Rect crop((resized.cols - geometry.height) / 2,
                  (resized.rows - geometry.width) / 2, 
                  geometry.height,
                  geometry.width);

    cv::cuda::GpuMat sample;
    if (resized.channels() == 3 && channels == 1)
        cv::cuda::cvtColor(resized, sample, cv::COLOR_BGR2GRAY);
    else if (resized.channels() == 4 && channels == 1)
        cv::cuda::cvtColor(resized, sample, cv::COLOR_BGRA2GRAY);
    else if (resized.channels() == 4 && channels == 3)
        cv::cuda::cvtColor(resized, sample, cv::COLOR_BGRA2BGR);
    else if (resized.channels() == 1 && channels == 3)
        cv::cuda::cvtColor(resized, sample, cv::COLOR_GRAY2BGR);
    else
        sample = resized;
   
    cv::cuda::GpuMat mfloat;
    if (channels == 3)
        sample.convertTo(mfloat, CV_32FC3, 1.0, -128);
    else
        sample.convertTo(mfloat, CV_32FC1, 1.0, -128);

    std::vector<cv::cuda::GpuMat> result;
    cv::cuda::split(mfloat, result);

    return copy_cuda_data()(result,
                            sample.channels(),
                            sample.rows,
                            sample.cols);
}

}
