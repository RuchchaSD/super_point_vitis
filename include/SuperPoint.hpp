#pragma once
#include <string>
#include <vector>

#include <memory>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <iostream>
#include <fstream>


#include <queue>
#include <atomic>

#include <glog/logging.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/math.hpp>

#define HW_SOFTMAX
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

DEF_ENV_PARAM(DEBUG_SUPERPOINT, "0");
DEF_ENV_PARAM(DUMP_SUPERPOINT, "0");
DEF_ENV_PARAM(DEBUG_THREADS, "0");



using namespace std;
using namespace cv;


namespace vitis {
namespace ai {
    
  struct SuperPointResult {
    size_t index;  // To keep track of the image order
    std::vector<std::pair<float, float>> keypoints;
    std::vector<std::vector<float>> descriptor;
    float scale_w;
    float scale_h;
  };

  class SuperPoint {
    public:
      enum class ImplType {
        SINGLE_THREADED,
        MULTI_THREADED
      };
      
      static std::unique_ptr<SuperPoint> create(const std::string& model_name, 
                                                ImplType impl_type = ImplType::MULTI_THREADED,
                                                int num_runners = 1);

    protected:
      explicit SuperPoint(const std::string& model_name);
      SuperPoint(const SuperPoint&) = delete;
      SuperPoint& operator=(const SuperPoint&) = delete;

    public:
      virtual ~SuperPoint();
      virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs) = 0;
      virtual SuperPointResult run(const cv::Mat& img) = 0;
      virtual size_t get_input_batch() = 0;
      virtual int getInputWidth() const = 0;
      virtual int getInputHeight() const = 0;
  };
}
}