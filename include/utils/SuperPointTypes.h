#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

// Data structures for queue-based processing
struct InputQueueItem {
    size_t index;
    std::string name;
    cv::Mat image;
};

struct SuperPointResult {
    size_t index;  // To keep track of the image order
    std::string name;
    cv::Mat img;
    std::vector<cv::KeyPoint> keypoints_cv;  // OpenCV KeyPoints
    cv::Mat descriptors_cv;  // OpenCV Mat descriptors
    float scale_w;
    float scale_h;
};

// Data structures for pipeline stages
struct DpuInferenceTask {
    size_t index;
    std::string name;
    cv::Mat img;
    std::vector<int8_t> input_data;
    float scale_w;
    float scale_h;
};

struct DpuInferenceResult {
    size_t index;
    std::string name;
    cv::Mat img;
    std::vector<int8_t> output_data1;
    std::vector<int8_t> output_data2;
    float scale_w;
    float scale_h;
    float scale1;
    float scale2;
};