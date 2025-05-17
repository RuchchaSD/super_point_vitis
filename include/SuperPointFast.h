// SuperPointFast.h
#pragma once

#include <string>
#include <vector>

#include <glog/logging.h>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <cstddef>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/math.hpp>

// Include the new modular header files
#include "utils/ThreadSafeQueue.h"
#include "utils/Sequencer.h"
#include "utils/SuperPointTypes.h"
#include "utils/SuperPointUtils.h"
#include "utils/YOLOSegmenterClient.h"  // Added for segmentation masks

#define HW_SOFTMAX
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

DEF_ENV_PARAM(DEBUG_SUPERPOINT, "0");
DEF_ENV_PARAM(DUMP_SUPERPOINT, "0");

// External global atomic parameters (defined in SuperPointUtils.h and SuperPointFast.cpp)
extern std::atomic<float> g_conf_thresh;

using namespace std;
using namespace cv;

static vector<vitis::ai::library::OutputTensor> sort_tensors(
    const vector<vitis::ai::library::OutputTensor>& tensors,
    vector<size_t>& chas) {
  vector<vitis::ai::library::OutputTensor> ordered_tensors;
  for (auto i = 0u; i < chas.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].channel == chas[i]) {
        ordered_tensors.push_back(tensors[j]);
        LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT))
          << "tensor name: " << tensors[j].name;
        break;
      }
  return ordered_tensors;
}

// Multi-threaded implementation
class SuperPointFast {
    public:
    SuperPointFast(const std::string& model_name, int num_threads);

    public:
    virtual ~SuperPointFast();
    virtual std::vector<ResultQueueItem> run(const std::vector<cv::Mat>& imgs);

    void run(ThreadSafeQueue<InputQueueItem>& input_queue, ThreadSafeQueue<ResultQueueItem>& output_queue);

    virtual size_t get_input_batch();
    virtual int getInputWidth() const;
    virtual int getInputHeight() const;

    private:
    void pre_process(const std::vector<cv::Mat>& input_images,
                    ThreadSafeQueue<DpuInferenceTask>& task_queue,
                    int start_idx, int end_idx);
    DpuInferenceTask pre_process_image(const cv::Mat& img, int idx);
    DpuInferenceTask pre_process_image(InputQueueItem inputItem);
    void dpu_inference(ThreadSafeQueue<DpuInferenceTask>& task_queue,
                        ThreadSafeQueue<DpuInferenceResult>& result_queue);
    void post_process(ThreadSafeQueue<DpuInferenceResult>& result_queue,
                    size_t thread_idx, size_t num_threads);

    ResultQueueItem process_result(const DpuInferenceResult& result);
    

    private:
    static const int NUM_DPU_RUNNERS = 2;  // Fixed number of DPU runners
    int num_threads_ = 2;  // Number of pre/post-processing threads
    std::mutex results_mutex_;  // Mutex for synchronizing results access
    std::thread pipeline_thread_;

    // Segmentation mask support
    std::unique_ptr<YOLOSegmenterClient> segmenter_;
    std::string segmenter_url_;
    bool use_segmentation_mask_;

    std::vector<std::unique_ptr<vitis::ai::DpuTask>> runners_;
    std::vector<ResultQueueItem> results_;
    std::vector<vitis::ai::library::InputTensor> input_tensors_;
    vector<size_t> chans_;

    int sWidth;
    int sHeight;
    size_t batch_;
    // scale0 is the scale factor for the input tensor
    float scale0;


    size_t channel1;
    size_t channel2;
    size_t outputH;
    size_t outputW;
    size_t output2H;
    size_t output2W;
    size_t outputSize1;
    size_t outputSize2;
};