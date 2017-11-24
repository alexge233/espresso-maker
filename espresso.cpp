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

void channel_split::operator()(caffe2::TensorCUDA & tensor,
                               const gpu_mat mfloat) const
{
    auto ptr = tensor.mutable_data<float>();
    size_t width = mfloat.cols * mfloat.rows;
    std::vector<cv::cuda::GpuMat> input_channels {
        cv::cuda::GpuMat(mfloat.rows, mfloat.cols, CV_32F, &ptr[0]),
        cv::cuda::GpuMat(mfloat.rows, mfloat.cols, CV_32F, &ptr[width]),
        cv::cuda::GpuMat(mfloat.rows, mfloat.cols, CV_32F, &ptr[width * 2])
    };
    cv::cuda::split(mfloat, input_channels);
}

void make_cuda_tensor::operator()(caffe2::TensorCUDA & tensor,
                                  const gpu_mat image,
                                  unsigned int channels,
                                  cv::Size geometry) const
{
    // resize image to the geometry needed (usually 224 * 224)
    cv::cuda::GpuMat resized;
    if (image.size() != geometry)
        cv::cuda::resize(image, resized, geometry);
    else
        resized = image;

    // crop the image if needed
    cv::Rect crop((resized.cols - geometry.height) / 2,
                  (resized.rows - geometry.width) / 2, 
                  geometry.height,
                  geometry.width);

    // convert colour to BGR
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

    auto ptr = tensor.mutable_data<float>();
    size_t width = mfloat.cols * mfloat.rows;
    std::vector<cv::cuda::GpuMat> input_channels {
        cv::cuda::GpuMat(mfloat.rows, mfloat.cols, CV_32F, &ptr[0]),
        cv::cuda::GpuMat(mfloat.rows, mfloat.cols, CV_32F, &ptr[width]),
        cv::cuda::GpuMat(mfloat.rows, mfloat.cols, CV_32F, &ptr[width * 2])
    };
    cv::cuda::split(mfloat, input_channels);
}

}
