#include <iostream>
#include "espresso.hpp"
#include <opencv2/opencv.hpp>

void save_to_disk(cv::cuda::GpuMat mat,
                  std::string name)
{
    cv::Mat tmp(mat);
    cv::imwrite(name, tmp);
}

int main()
{
    // load some image from file -- edit the filename!
    cv::Mat image = cv::imread("../example.jpeg", CV_LOAD_IMAGE_COLOR);
   
    // upload image to the GPU
    cv::cuda::GpuMat gpu_mat(image);

    // pre-process:
    //  1. resize to the blob input geometry (usually 224 * 224)
    //     it is also cropped if needed
    auto resized   = espresso::preprocess()(gpu_mat);
    save_to_disk(resized, "preprocessed.png");

    //  2. convert to BGR - 3 Channel
    auto converted = espresso::convert()(resized);
    save_to_disk(converted, "converted.png");

    //  3. turn into a float matrix
    auto mfloat   = espresso::make_float()(converted);
    save_to_disk(mfloat, "float.png");

    //  4. allocate a tensor - make sure to set CHW format correctly
    caffe2::TensorCUDA tensor;
    std::vector<caffe2::TIndex> dims({1, mfloat.channels(), mfloat.rows, mfloat.cols});
    tensor.Resize(dims);

    //  5. populate it with the float BGR data now
    espresso::make_cuda_tensor()(tensor, mfloat);

    //  Your tensor now has the BGR float data in CHW format.
    //  You can use it by feeding it into the "input" blob of a network

    return 0;
}
