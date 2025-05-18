//SuperPointTypes.h
#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

// Input queue item for the feature extractor pipeline
struct InputQueueItem {
    InputQueueItem() : index(0), timestamp(0.0) {}
    InputQueueItem(
        int idx, double ts, 
        const std::string &fname, const cv::Mat &img)
        : index(idx), timestamp(ts), filename(fname), image(img) {}
    
    int index;              // Image index in sequence
    double timestamp;       // Timestamp of the image
    std::string filename;   // Original filename path
    cv::Mat image;          // Image data
};

// Result queue item containing extracted features
struct ResultQueueItem {
    ResultQueueItem() : index(0), timestamp(0.0) {}
    ResultQueueItem(
        int idx, double ts, 
        const std::string &fname, const cv::Mat &img)
        : index(idx), timestamp(ts), filename(fname), image(img) {}

    int index;                      // Image index in sequence
    double timestamp;               // Timestamp of the image
    std::string filename;           // Original filename path
    cv::Mat image;                  // Image data
    std::vector<cv::KeyPoint> keypoints;  // Extracted keypoints
    cv::Mat descriptors;            // Feature descriptors
    std::vector<int> lappingArea;   // For compatibility with ORB extractor
};


// Data structures for pipeline stages
struct DpuInferenceTask {
    int index;
    double timestamp;
    std::string filename;
    cv::Mat img;
    std::vector<int8_t> input_data;
    float scale_w;
    float scale_h;
    std::unique_ptr<std::future<cv::Mat>> mask_future;  // Future for asynchronous mask retrieval
    
    // Add move constructor
    DpuInferenceTask() = default;
    
    // Delete copy constructor and assignment operator
    DpuInferenceTask(const DpuInferenceTask&) = delete;
    DpuInferenceTask& operator=(const DpuInferenceTask&) = delete;
    
    // Add move constructor and assignment operator
    DpuInferenceTask(DpuInferenceTask&& other) noexcept
        : index(other.index),
          timestamp(other.timestamp),
          filename(std::move(other.filename)),
          img(std::move(other.img)),
          input_data(std::move(other.input_data)),
          scale_w(other.scale_w),
          scale_h(other.scale_h),
          mask_future(std::move(other.mask_future))
    {}
    
    DpuInferenceTask& operator=(DpuInferenceTask&& other) noexcept {
        if (this != &other) {
            index = other.index;
            timestamp = other.timestamp;
            filename = std::move(other.filename);
            img = std::move(other.img);
            input_data = std::move(other.input_data);
            scale_w = other.scale_w;
            scale_h = other.scale_h;
            mask_future = std::move(other.mask_future);
        }
        return *this;
    }
};

struct DpuInferenceResult {
    int index;
    double timestamp;
    std::string filename;
    cv::Mat img;
    std::vector<int8_t> output_data1;
    std::vector<int8_t> output_data2;
    float scale_w;
    float scale_h;
    float scale1;
    float scale2;
    // cv::Mat mask;  // Store the actual mask instead of future
    std::unique_ptr<std::future<cv::Mat>> mask_future;  // Future for asynchronous mask retrieval
    
    // Add move constructor
    DpuInferenceResult() = default;
    
    // Delete copy constructor and assignment operator
    DpuInferenceResult(const DpuInferenceResult&) = delete;
    DpuInferenceResult& operator=(const DpuInferenceResult&) = delete;
    
    // Add move constructor and assignment operator
    DpuInferenceResult(DpuInferenceResult&& other) noexcept
        : index(other.index),
          timestamp(other.timestamp),
          filename(std::move(other.filename)),
          img(std::move(other.img)),
          output_data1(std::move(other.output_data1)),
          output_data2(std::move(other.output_data2)),
          scale_w(other.scale_w),
          scale_h(other.scale_h),
          scale1(other.scale1),
          scale2(other.scale2),
          mask_future(std::move(other.mask_future))
    {}
    
    DpuInferenceResult& operator=(DpuInferenceResult&& other) noexcept {
        if (this != &other) {
            index = other.index;
            timestamp = other.timestamp;
            filename = std::move(other.filename);
            img = std::move(other.img);
            output_data1 = std::move(other.output_data1);
            output_data2 = std::move(other.output_data2);
            scale_w = other.scale_w;
            scale_h = other.scale_h;
            scale1 = other.scale1;
            scale2 = other.scale2;
            mask_future = std::move(other.mask_future);
        }
        return *this;
    }
};

// SuperPointResult is now deprecated, use ResultQueueItem instead
// Kept here for reference if needed during migration
struct SuperPointResult {
    int index;
    double timestamp;
    std::string filename;
    cv::Mat img;
    std::vector<cv::KeyPoint> keypoints_cv;
    cv::Mat descriptors_cv;
    float scale_w;
    float scale_h;
    
    // Helper method to convert to ResultQueueItem
    ResultQueueItem toResultQueueItem() const {
        ResultQueueItem result;
        result.index = index;
        result.timestamp = timestamp;
        result.filename = filename;
        result.image = img;
        result.keypoints = keypoints_cv;
        result.descriptors = descriptors_cv;
        return result;
    }
};