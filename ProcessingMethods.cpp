#include "ProcessingMethods.hpp"

cv::Mat ProcessingMethods::rollingGuidanceFilterFrame(cv::Mat frame) {
    cv::Mat result;
    cv::ximgproc::rollingGuidanceFilter(frame, result, 3, 25, 4);
    return result;
}

cv::Mat ProcessingMethods::weightedMedianFilterFrame(cv::Mat frame) {
    cv::Mat result;
    // используем ту же матрицу как веса
    cv::ximgproc::weightedMedianFilter(frame, frame, result, 5, 0.5);
    return result;
}

cv::Mat ProcessingMethods::niBlackThresholdFrame(cv::Mat frame) {
    cv::Mat gray, result;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    // метод: 0 = BINARIZATION_NIBLACK, 1 = BINARIZATION_SAUVOLA
    cv::ximgproc::niBlackThreshold(gray, result, 255, cv::THRESH_BINARY, 15, 0.2, 0);
    return result;
}

cv::Mat ProcessingMethods::anisotropicDiffusionFrame(cv::Mat frame) {
    cv::Mat result;
    // 10 итераций, alpha = 0.1, k = 0.01
    cv::ximgproc::anisotropicDiffusion(frame, result, 10, 0.1, 0.01);
    return result;
}

cv::Mat ProcessingMethods::GrayGaussprocessFrame(cv::Mat frame) {
    cv::Mat grayImage;
    cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);
    return blurredImage;
}

cv::Mat ProcessingMethods::GaussprocessFrame(cv::Mat frame) {
    cv::Mat blurredImage;
    cv::GaussianBlur(frame, blurredImage, cv::Size(5, 5), 0);
    return blurredImage;
}

cv::Mat ProcessingMethods::MedianprocessFrame(cv::Mat frame) {
    cv::Mat blurredImage;
    cv::medianBlur(frame, blurredImage, 5);
    return blurredImage;
}

cv::Mat ProcessingMethods::bilateralprocessFrame(cv::Mat frame) {
    cv::Mat blurredImage;
    cv::bilateralFilter(frame, blurredImage, 9, 75, 75); // хуй знает что делают аргументы
    return blurredImage;
}

cv::Mat ProcessingMethods::fastNlprocessFrame(cv::Mat frame) {
    cv::Mat blurredImage;
    cv::fastNlMeansDenoisingColored(frame, blurredImage, 10, 10, 21); // хуй знает что делают аргументы
    // у этой функции кстати есть улучшение связанное с обработкой нескольких кадров сразу, может быть полезно
    // ссылка на документацию https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
    return blurredImage;
}