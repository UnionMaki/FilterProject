#pragma once
#include "VideoFrame.hpp"
#include "FrameParametres.hpp"
#include <string>

class ProcessedVideo {
private:
    friend class TestingMetods;
    double fps;
    std::string process_type = "default";
    std::vector<VideoFrame> Framelist;
    FrameParametres settings;

public:

    int processChunk(const std::string& videoFile, cv::Mat(*filter)(cv::Mat));

    void processChunk(const std::string& videoFile, void (*filter)(ProcessedVideo*, cv::Mat));

    void PlayVideo();

};