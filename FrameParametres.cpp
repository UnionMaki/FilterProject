#include "FrameParametres.hpp"

std::pair<int, int> FrameParametres::getScreenResolution() {

    cv::namedWindow("temp", cv::WINDOW_NORMAL);
    cv::setWindowProperty("temp", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    
    cv::Rect screenRect = cv::getWindowImageRect("temp");
    cv::destroyWindow("temp");
    
    return {screenRect.width, screenRect.height};
}

void FrameParametres::SetPos() {
    auto [xscreen, yscreen] = getScreenResolution();

    height = (int)yscreen / 2;
    width = int(height * (4./3.));

    x = (int)xscreen / 2 - width - gap / 2;
    y = (int)yscreen / 4;
}

FrameParametres::FrameParametres() {
    SetPos();
}