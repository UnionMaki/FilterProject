#pragma once
#include <iostream>
#include "MyOpenCV.hpp"

class FrameParametres {
private:
    friend class ProcessedVideo;
    int x = 300;
    int y = 300;
    int width = 2400;
    int height = 1800;
    int gap = 20;

public:
    std::pair<int, int> getScreenResolution(); 

    void SetPos();

    FrameParametres();
};