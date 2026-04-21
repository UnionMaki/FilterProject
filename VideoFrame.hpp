#pragma once
#include "MyOpenCV.hpp"

class VideoFrame {
public:
    cv::Mat original;
    cv::Mat processed;
};