#pragma once
#include "MyOpenCV.hpp"

class ProcessingMethods {
public:

    static cv::Mat rollingGuidanceFilterFrame(cv::Mat frame);

    static cv::Mat weightedMedianFilterFrame(cv::Mat frame);

    static cv::Mat niBlackThresholdFrame(cv::Mat frame);

    static cv::Mat anisotropicDiffusionFrame(cv::Mat frame);

    static cv::Mat GrayGaussprocessFrame(cv::Mat frame);

    static cv::Mat GaussprocessFrame(cv::Mat frame);

    static cv::Mat MedianprocessFrame(cv::Mat frame);

    static cv::Mat bilateralprocessFrame(cv::Mat frame);

    static cv::Mat fastNlprocessFrame(cv::Mat frame);

};