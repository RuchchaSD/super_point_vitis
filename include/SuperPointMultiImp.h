#pragma once
#include <thread>

#include "utils.h"
#include "SuperPoint.hpp"
#include "Threadpool.h"

namespace vitis {
namespace ai {

// Multi-threaded implementation
class SuperPointMultiImp : public SuperPoint {
    public:
     SuperPointMultiImp(const std::string& model_name, int num_threads);
   
    public:
     virtual ~SuperPointMultiImp();
     virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs) override;
     virtual SuperPointResult run(const cv::Mat& img) override;
     void run(ThreadSafeQueue<cv::Mat>& imageQueue, ThreadSafeQueue<SuperPointResult>& result_queue, const bool& stop);
     virtual size_t get_input_batch() override;
     virtual int getInputWidth() const override;
     virtual int getInputHeight() const override;
   
    private:
     DpuInferenceTask pre_process(const PreProcessTask& input_image);
     DpuInferenceResult dpu_inference(const DpuInferenceTask& task, size_t runnerIdx);
     SuperPointResult post_process(const DpuInferenceResult& result);
   
    private:
     static const int NUM_DPU_RUNNERS = 4;  // Fixed number of DPU runners
     int num_threads_;  // Number of pre/post-processing threads
     std::mutex results_mutex_;  // Mutex for synchronizing results access
   
     std::vector<std::unique_ptr<vitis::ai::DpuTask>> runners_;
     std::vector<SuperPointResult> results_;
     std::vector<vitis::ai::library::InputTensor> input_tensors_;
     vector<size_t> chans_;
   
     int sWidth;
     int sHeight;
     size_t batch_;
   
     size_t channel1;
     size_t channel2;
     size_t outputH;
     size_t outputW;
     size_t output2H;
     size_t output2W;
     float conf_thresh;
     size_t outputSize1;
     size_t outputSize2;
};

} // namespace ai
} // namespace vitis
