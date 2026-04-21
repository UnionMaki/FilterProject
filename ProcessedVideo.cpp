#include "ProcessedVideo.hpp"

int ProcessedVideo::processChunk(const std::string& videoFile, cv::Mat(*filter)(cv::Mat)) { //оставил пока одно видео, а не вектор
    cv::VideoCapture cap(videoFile, cv::CAP_FFMPEG);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot connect to stream\n";
        return -1;
    }

    fps = cap.get(cv::CAP_PROP_FPS);
    cv::Mat frame;

    cv::namedWindow("processed", cv::WINDOW_NORMAL);
    double delay = 1000 / fps;

    while (cap.read(frame)) {
        cv::imshow("orig", frame);
        cv::imshow("processed", filter(frame));
        cv::resizeWindow("orig", settings.x, settings.y);
        cv::resizeWindow("processed", settings.x, settings.y);
        cv::moveWindow("orig", settings.width, settings.height);
        cv::moveWindow("processed", settings.width + settings.x + settings.gap, settings.height);

        if (cv::waitKey(delay) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

void ProcessedVideo::processChunk(const std::string& videoFile, void (*filter)(ProcessedVideo*, cv::Mat)) { //оставил пока одно видео, а не вектор
    cv::VideoCapture cap(videoFile);

    fps = cap.get(cv::CAP_PROP_FPS);
    cv::Mat frame;

    double delay = 1000 / fps;

    while (cap.read(frame)) {
        filter(this, frame);
    }

    cap.release();
    cv::destroyAllWindows();
}

void ProcessedVideo::PlayVideo() {
    if (Framelist.empty()) {
        std::cout << "Нет кадров для отображения" << std::endl;
        return;
    }
    cv::namedWindow("orig", cv::WINDOW_NORMAL);
    cv::namedWindow(process_type, cv::WINDOW_NORMAL);
    cv::resizeWindow("orig", settings.width, settings.height);
    cv::resizeWindow(process_type, settings.width, settings.height);
    cv::moveWindow("orig", settings.x, settings.y);
    cv::moveWindow(process_type, settings.width + settings.x + settings.gap, settings.y);

    double delay = 1000 / fps;

    for (auto& frame : Framelist) {
        cv::imshow("orig", frame.original);
        cv::imshow(process_type, frame.processed);
        if (cv::waitKey(delay) == 27) break; // ESC для выхода
    }

    cv::destroyAllWindows();
    cv::waitKey(0);
}
