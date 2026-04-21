#pragma once
#include "MyOpenCV.hpp"
#include "VideoFrame.hpp"
#include <string>
#include "ProcessedVideo.hpp"

class TestingMetods {
public:
static void rollingGuidanceProcessFrame(ProcessedVideo* obj, cv::Mat frame);
static void weightedMedianFilterFrame(ProcessedVideo* obj, cv::Mat frame);
static void niBlackThresholdFrame(ProcessedVideo* obj, cv::Mat frame);
static void anisotropicDiffusionFrame(ProcessedVideo* obj, cv::Mat frame);
static void bilateralSharpprocessFrame(ProcessedVideo* obj, cv::Mat frame);
static void GrayGaussprocessFrame(ProcessedVideo* obj, cv::Mat frame);
static void GaussprocessFrame(ProcessedVideo* obj, cv::Mat frame);
static void MedianprocessFrame(ProcessedVideo* obj, cv::Mat frame);
static void bilateralprocessFrame(ProcessedVideo* obj, cv::Mat frame);
static void fastNlprocessFrame(ProcessedVideo* obj, cv::Mat frame);
static void bilateralSharpHeavy(ProcessedVideo* obj, cv::Mat frame);
static void bilateralSharpSubtle(ProcessedVideo* obj, cv::Mat frame);
static void bilateralSharpCross(ProcessedVideo* obj, cv::Mat frame);
static void bilateralSharpEmboss(ProcessedVideo* obj, cv::Mat frame);
static void bilateralSharpChromatic(ProcessedVideo* obj, cv::Mat frame);
static void bilateralSharpRing(ProcessedVideo* obj, cv::Mat frame);
static void guidedSharpHeavy(ProcessedVideo* obj, cv::Mat frame);
static void rollingSharpSubtle(ProcessedVideo* obj, cv::Mat frame);
static void anisotropicSharpCross(ProcessedVideo* obj, cv::Mat frame);
static void l0SmoothSharpEmboss(ProcessedVideo* obj, cv::Mat frame);
static void fgsSharpChromatic(ProcessedVideo* obj, cv::Mat frame);
static void dtSharpRing(ProcessedVideo* obj, cv::Mat frame);
static void wmfSharpDiamond(ProcessedVideo* obj, cv::Mat frame);
static void jointBilateralSharpLarge(ProcessedVideo* obj, cv::Mat frame);

};