#include <string>
#include <iostream>
#include "ProcessedVideo.hpp"
#include "TestingMetods.hpp"
#include "ProcessingMethods.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "arguments required" << std::endl;
        return 1;
    }
    else {
        ProcessedVideo SVO;
        ProcessedVideo SVO_TEST;
        if (std::string(argv[1]) != "-f" && std::string(argv[1]) != "-s")
            std::cout << "wrong arguments(2)" << std::endl;
        if (std::string(argv[1]) == "-f"){
            SVO_TEST.processChunk(argv[2], TestingMetods::bilateralSharpCross);
            SVO_TEST.PlayVideo();
        }
        if (std::string(argv[1]) == "-s")
            SVO.processChunk(argv[2], ProcessingMethods::bilateralprocessFrame);
        return 0;
    }
}
