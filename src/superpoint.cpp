/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// superpoint.cpp

#include "superpoint.hpp"

#include <glog/logging.h>
#include <memory>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>
#include <atomic>

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

namespace vitis {
namespace ai {

// Thread-safe queue implementation
template <typename T>
class ThreadSafeQueue {
 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;
  bool shutdown_;

 public:
  ThreadSafeQueue() : shutdown_(false) {}

  void enqueue(const T& item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(item);
    }
    cond_var_.notify_one();
  }

  bool dequeue(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [this]() { return !queue_.empty() || shutdown_; });
    if (shutdown_ && queue_.empty()) {
      return false;
    }
    item = queue_.front();
    queue_.pop();
    return true;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cond_var_.notify_all();
  }
};


struct PreProcessTask {
  int index;
  cv::Mat image;;
};
  
// Data structures for pipeline stages
struct DpuInferenceTask {
  size_t index;
  std::vector<int8_t> input_data;
  float scale_w;
  float scale_h;
};

struct DpuInferenceResult {
  size_t index;
  std::vector<int8_t> output_data1;
  std::vector<int8_t> output_data2;
  float scale_w;
  float scale_h;
  float scale1;
  float scale2;
};

// Common utility functions for both implementations
inline void L2_normalization(const int8_t* input, float scale, int channel, int group, float* output) {
#ifdef ENABLE_NEON
  // NEON vectorized L2 normalization (from optimized implementation)
  const size_t blk = 32;
  for (size_t g = 0; g < group; ++g) {
    const int8_t* src = input + g * channel;
    float32x4_t sumv = vdupq_n_f32(0.f);
    
    for (size_t c = 0; c < channel; c += blk) {
      int8x16_t v0 = vld1q_s8(src + c);
      int8x16_t v1 = vld1q_s8(src + c + 16);
      int16x8_t s0 = vmovl_s8(vget_low_s8(v0));
      int16x8_t s1 = vmovl_s8(vget_high_s8(v0));
      int16x8_t s2 = vmovl_s8(vget_low_s8(v1));
      int16x8_t s3 = vmovl_s8(vget_high_s8(v1));
      
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))));
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))));
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))));
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))));
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))));
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))));
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))));
      sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))));
    }
    
    // Using fixed indices instead of a loop variable
    float sum = vgetq_lane_f32(sumv, 0) + vgetq_lane_f32(sumv, 1) + 
                vgetq_lane_f32(sumv, 2) + vgetq_lane_f32(sumv, 3);
    
    float norm = 1.f / std::sqrt(sum) * scale;
    float32x4_t nrm = vdupq_n_f32(norm);
    
    for (size_t c = 0; c < channel; c += 16) {
      int8x16_t v = vld1q_s8(src + c);
      int16x8_t lo = vmovl_s8(vget_low_s8(v));
      int16x8_t hi = vmovl_s8(vget_high_s8(v));
      
      float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), nrm);
      float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), nrm);
      float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), nrm);
      float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), nrm);
      
      vst1q_f32(output + g * channel + c + 0, f0);
      vst1q_f32(output + g * channel + c + 4, f1);
      vst1q_f32(output + g * channel + c + 8, f2);
      vst1q_f32(output + g * channel + c + 12, f3);
    }
  }
#else
  // Scalar L2 normalization
  for (int i = 0; i < group; ++i) {
    float sum = 0.0;
    for (int j = 0; j < channel; ++j) {
      int pos = i * channel + j;
      float temp = input[pos] * scale;
      sum += temp * temp;
    }
    float var = sqrt(sum);
    for (int j = 0; j < channel; ++j) {
      int pos = i * channel + j;
      output[pos] = (input[pos] * scale) / var;
    }
  }
#endif
}

// Bilinear interpolation helper function
inline float bilinear_interpolation(float v00, float v01, float v10, float v11,
                                   int x0, int y0, int x1, int y1,
                                   float x, float y, bool border_check) {
  if (border_check) {
    // Out of bounds check
    if (x0 < 0 || y0 < 0 || x1 < 0 || y1 < 0) {
      return 0;
    }
  }
  
  float dx = (x - x0) / static_cast<float>(x1 - x0);
  float dy = (y - y0) / static_cast<float>(y1 - y0);
  
  float val = (1 - dx) * (1 - dy) * v00 +
              dx * (1 - dy) * v01 +
              (1 - dx) * dy * v10 +
              dx * dy * v11;
  
  return val;
}

// Optimized bilinear sampling for descriptor maps
inline void bilinear_sample(const float* map, size_t h, size_t w, size_t ch,
                           const std::vector<std::pair<float, float>>& pts,
                           std::vector<std::vector<float>>& descs) {
  descs.resize(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    int x0 = floor(pts[i].first / 8.f);
    int y0 = floor(pts[i].second / 8.f);
    int x1 = std::min<int>(x0 + 1, w - 1);
    int y1 = std::min<int>(y0 + 1, h - 1);
    float dx = pts[i].first / 8.f - x0;
    float dy = pts[i].second / 8.f - y0;
    float w00 = (1 - dx) * (1 - dy);
    float w01 = dx * (1 - dy);
    float w10 = (1 - dx) * dy;
    float w11 = dx * dy;
    
    descs[i].resize(ch);
    float norm = 0.f;
    for (size_t c = 0; c < ch; ++c) {
      float val = map[c + ch * (y0 * w + x0)] * w00 +
                 map[c + ch * (y0 * w + x1)] * w01 +
                 map[c + ch * (y1 * w + x0)] * w10 +
                 map[c + ch * (y1 * w + x1)] * w11;
      descs[i][c] = val;
      norm += val * val;
    }
    
    // Normalize the descriptor
    norm = 1.f / std::sqrt(norm);
    for (auto& v : descs[i]) v *= norm;
  }
}

// Optimized NMS implementation from the optimized code
inline void nms_fast(const std::vector<int>& xs, const std::vector<int>& ys,
                    const std::vector<float>& score, int w, int h,
                    std::vector<size_t>& keep) {
  const int radius = 4;
  std::vector<int> grid(w * h, 0);
  std::vector<size_t> order(xs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return score[a] > score[b]; });
  for (auto idx : order) {
    int x = xs[idx], y = ys[idx];
    if (x < radius || x >= w - radius || y < radius || y >= h - radius) continue;
    bool skip = false;
    for (int i = -radius; i <= radius && !skip; ++i)
      for (int j = -radius; j <= radius; ++j)
        if (grid[(y + i) * w + (x + j)] == 1) { skip = true; break; }
    if (!skip) {
      keep.push_back(idx);
      for (int i = -radius; i <= radius; ++i)
        for (int j = -radius; j <= radius; ++j)
          grid[(y + i) * w + (x + j)] = 1;
    }
  }
}

// Helper function to match existing interface with the optimized NMS
inline void nms_fast(const std::vector<int>& xs, const std::vector<int>& ys,
                    const std::vector<float>& score, std::vector<size_t>& keep_inds, int w, int h) {
  nms_fast(xs, ys, score, w, h, keep_inds);
}

vector<vector<float>> grid_sample(const float* desc_map, const vector<pair<float, float>>& coarse_pts,
                                 const size_t channel, const size_t outputH, const size_t outputW) {
  vector<vector<float>> descs(coarse_pts.size());
  for (size_t i = 0; i < coarse_pts.size(); ++i) {
    float x = (coarse_pts[i].first + 1) / 8 - 0.5;
    float y = (coarse_pts[i].second + 1) / 8 - 0.5;
    int xmin = floor(x);
    int ymin = floor(y);
    int xmax = xmin + 1;
    int ymax = ymin + 1;

    xmin = std::max(0, std::min(xmin, static_cast<int>(outputW) - 1));
    xmax = std::max(0, std::min(xmax, static_cast<int>(outputW) - 1));
    ymin = std::max(0, std::min(ymin, static_cast<int>(outputH) - 1));
    ymax = std::max(0, std::min(ymax, static_cast<int>(outputH) - 1));

    // Bilinear interpolation
    {
      float divisor = 0.0;
      for (size_t j = 0; j < channel; ++j) {
        float value = bilinear_interpolation(
            desc_map[j + (ymin * outputW + xmin) * channel],
            desc_map[j + (ymin * outputW + xmax) * channel],
            desc_map[j + (ymax * outputW + xmin) * channel],
            desc_map[j + (ymax * outputW + xmax) * channel], xmin, ymin, xmax, ymax, x, y, false);
        divisor += value * value;
        descs[i].push_back(value);
      }
      for (size_t j = 0; j < channel; ++j) {
        descs[i][j] /= sqrt(divisor);  // L2 normalize
      }
    }
  }
  return descs;
}

// Single-threaded implementation
class SuperPointSingleImp : public SuperPoint {
 public:
  SuperPointSingleImp(const std::string& model_name);
  virtual ~SuperPointSingleImp();
  
  virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs) override;
  virtual size_t get_input_batch() override;
  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;

 private:
  void set_input(vitis::ai::library::InputTensor& tensor, float mean, float scale, vector<Mat>& img);
  void superpoint_run(const vector<cv::Mat>& input_images);
  bool process_outputs(size_t count);

 private:
  std::unique_ptr<vitis::ai::DpuTask> task_;
  vector<vitis::ai::library::InputTensor> inputs_;
  vector<vitis::ai::library::OutputTensor> outputs_;
  std::vector<SuperPointResult> results_;

  int sWidth;
  int sHeight;
  size_t batch_;
  vector<size_t> chans_;

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

// Multi-threaded implementation
class SuperPointMultiImp : public SuperPoint {
 public:
  SuperPointMultiImp(const std::string& model_name, int num_threads);

 public:
  virtual ~SuperPointMultiImp();
  virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs) override;
  void run(ThreadSafeQueue<cv::Mat>& imageQueue, ThreadSafeQueue<SuperPointResult>& result_queue,const bool& stop);
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

SuperPoint::SuperPoint(const std::string& model_name) {}

SuperPoint::~SuperPoint() {}

std::unique_ptr<SuperPoint> SuperPoint::create(const std::string& model_name, 
                                              ImplType impl_type,
                                              int num_runners) {
  if (impl_type == ImplType::SINGLE_THREADED) {
    return std::unique_ptr<SuperPointSingleImp>(new SuperPointSingleImp(model_name));
  } else {
    return std::unique_ptr<SuperPointMultiImp>(new SuperPointMultiImp(model_name, num_runners));
  }
}

// Single-threaded implementation
SuperPointSingleImp::SuperPointSingleImp(const std::string& model_name): SuperPoint(model_name) {
  task_ = vitis::ai::DpuTask::create(model_name);
  inputs_ = task_->getInputTensor(0u);
  sWidth = inputs_[0].width;
  sHeight = inputs_[0].height;
  batch_ = inputs_[0].batch;
  chans_ = {65,256};
  outputs_ = sort_tensors(task_ -> getOutputTensor(0u), chans_);
  channel1 = outputs_[0].channel;
  channel2 = outputs_[1].channel;
  outputH = outputs_[0].height;
  outputW = outputs_[0].width;
  output2H = outputs_[1].height;
  output2W = outputs_[1].width;
  conf_thresh = 0.015;

  LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
    << "tensor1 info : " << outputs_[0].height << " " << outputs_[0].width  << " " << outputs_[0].channel << endl
    << "tensor2 info : " << outputs_[1].height << " " << outputs_[1].width  << " " << outputs_[1].channel << endl;

  outputSize1 = outputs_[0].channel * outputs_[0].height * outputs_[0].width;
  outputSize2 = outputs_[1].channel * outputs_[1].height * outputs_[1].width;
}

SuperPointSingleImp::~SuperPointSingleImp() {}

void SuperPointSingleImp::set_input(vitis::ai::library::InputTensor& tensor, float mean, float scale, vector<Mat>& img) {
  float scale0 = vitis::ai::library::tensor_scale(tensor);
  size_t isize = tensor.size / tensor.batch;
  __TIC__(RESIZE)
  for (size_t i = 0; i < img.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT))
      << "batch " << i << endl
      << "img info(h,w): " << img[i].rows << " " << img[i].cols << endl
      << "dpu info(h,w): " << sHeight << " " <<  sWidth << endl
      << "scale: " << scale0 << " size: " << isize << endl;
    Mat mat;
    if (img[i].rows == sHeight && img[i].cols == sWidth) {
      mat = img[i];
    } else {
      resize(img[i], mat, cv::Size(sWidth, sHeight));
    }
  __TOC__(RESIZE)

  __TIC__(SET_IMG)
    cv::Mat gray_img;
    cv::cvtColor(mat, gray_img, cv::COLOR_BGR2GRAY);
    int8_t* input_ptr = (int8_t*)tensor.get_data(i);
    for (size_t j = 0; j < isize; ++j) {
      input_ptr[j] = static_cast<int8_t>((gray_img.data[j] - mean) * scale0 * scale);
    }
  __TOC__(SET_IMG)
    if (ENV_PARAM(DUMP_SUPERPOINT)) {
      ofstream fout("fin_"+to_string(i)+".bin", ios::binary);
      fout.write((char*)input_ptr, sWidth*sHeight);
      fout.close();
      LOG(INFO) << "The input scale is : " << scale0;
    }
  }
}

bool SuperPointSingleImp::process_outputs(size_t count) {
  results_.clear();
  for (size_t n = 0; n < count; ++n) {
    SuperPointResult result_;
    result_.index = n;
    int8_t* out1 = (int8_t*)outputs_[0].get_data(n);
    int8_t* out2 = (int8_t*)outputs_[1].get_data(n);

    float scale1 = vitis::ai::library::tensor_scale(outputs_[0]);
    float scale2 = vitis::ai::library::tensor_scale(outputs_[1]);

    if(ENV_PARAM(DUMP_SUPERPOINT)) {
      ofstream ofs ("out1.bin", ios::binary);
      ofs.write((char*)out2, outputSize1);
      ofs.close();
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
      << "the scales: " << scale1 << " " << scale2 << endl;
    vector<float> output1(outputSize1);
    __TIC__(SOFTMAX)
#ifndef HW_SOFTMAX
    for (int i=0; i<outputH*outputW; ++i) {
      float sum{0.0f};
      int pos = i*channel1;
      for (int j=0; j<channel1; ++j){
        output1[pos + j] = std::exp(out1[j + pos]*scale1);
        sum += output1[pos + j];
      }
      for (int j=0; j<channel1; ++j){
        output1[pos+j] /= sum;
      }
    }
#else
    vitis::ai::softmax(out1, scale1, channel1, outputH*outputW, output1.data());
#endif
    __TOC__(SOFTMAX)

    __TIC__(HEATMAP)
    int reduced_size = (channel1-1)*outputH*outputW;
    vector<float> heatmap(reduced_size);
    // remove heatmap[-1,:,:]
    for (size_t i = 0; i < outputH*outputW; i++) {
      memcpy(heatmap.data()+i*(channel1-1), output1.data()+i*channel1, sizeof(float)*(channel1-1));
    }
    __TOC__(HEATMAP)
    
    vector<float> tmp;
    tmp.reserve(reduced_size);
    vector<int> xs, ys;
    vector<size_t> keep_inds;
    vector<float> ptscore;
    __TIC__(SORT)
    for (size_t m = 0u; m < outputH; ++m){
      for (size_t i = 0u; i < 8; ++i){
        for (size_t n = 0u; n < outputW; ++n){
          for (size_t j = 0u; j < 8; ++j){
            tmp.push_back(heatmap.at(i*8 + j + (m*outputW + n)*64)); //transpose heatmap
            if (tmp.back() > conf_thresh){
              ys.push_back(m*8+i);
              xs.push_back(n*8+j);
              ptscore.push_back(tmp.back());
            }
          }
        }
      }
    }
    __TOC__(SORT)

    __TIC__(NMS)
    nms_fast(xs, ys, ptscore, keep_inds, sWidth, sHeight);
    __TOC__(NMS)

    __TIC__(L2_NORMAL)
    vector<float> output2(outputSize2);
    LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
      << "L2 normal: channel " << channel2 << " h: " << outputH << " w: " << outputW;
    L2_normalization(out2, scale2, channel2, output2H*output2W, output2.data());
    __TOC__(L2_NORMAL)

    __TIC__(DESC)
    for (size_t i = 0; i < keep_inds.size(); ++i) {
        std::pair<float, float> pt;
        pt.first = float(xs[keep_inds[i]]);
        pt.second = float(ys[keep_inds[i]]);
        result_.keypoints.push_back(pt);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
      << "keypoints size: " << result_.keypoints.size();
    result_.descriptor = grid_sample(output2.data(), result_.keypoints, channel2, output2H, output2W);
    __TOC__(DESC)

    if(ENV_PARAM(DEBUG_SUPERPOINT)) {
      if (result_.descriptor.size() > 0) {
        cout<<"desc of pt0 :"<<endl;
        for (int i=0; i< 64; ++i){
            if(i%8==0){ cout<<endl;}
            cout<<result_.descriptor[0][i]<<"  ";
        }
        cout << endl;
        cout<<"desc of pt1 :"<<endl;
        for (int i=0; i< 64; ++i){
            if(i%8==0){ cout<<endl;}
            cout<<result_.descriptor[1][i]<<"  ";
        }
        cout << endl;
      }
    }
    results_.push_back(result_);
  }
  return true;
}

void SuperPointSingleImp::superpoint_run(const std::vector<cv::Mat>& input_image) {
  auto input_tensor = inputs_[0];
  auto group = input_image.size() / batch_;
  auto rest = input_image.size() % batch_;
  auto img_iter = input_image.begin();
  auto img_end = img_iter;
  if (rest > 0) group += 1;
  size_t count = batch_;
  
  for (size_t g = 0; g < group; ++g) {
    __TIC__(PREPROCESS)
    size_t dist = std::distance(img_iter, input_image.end());
    if (dist > batch_)
      img_end += batch_;
    else {
      count = std::distance(img_iter, input_image.end());
      img_end = input_image.end();
    }
    vector<Mat> imgs(img_iter, img_end);
    img_iter = img_end;
    // set mean=0, scale=1/255.0
    set_input(input_tensor, 0, 0.00392157, imgs);
    __TOC__(PREPROCESS)

    __TIC__(DPU_RUN)
    task_->run(0u);
    __TOC__(DPU_RUN)

    __TIC__(POSTPROCESS)
    process_outputs(count);
    __TOC__(POSTPROCESS)
    
    for (size_t j = 0; j < count; ++j) {
      results_[j].scale_w = imgs[j].cols/(float)sWidth;
      results_[j].scale_h = imgs[j].rows/(float)sHeight;
    }
  }
}

std::vector<SuperPointResult> SuperPointSingleImp::run(const std::vector<cv::Mat>& imgs) {
  results_.clear();
  superpoint_run(imgs);
  return results_;
}

size_t SuperPointSingleImp::get_input_batch() { return task_->get_input_batch(0, 0); }
int SuperPointSingleImp::getInputWidth() const { return task_->getInputTensor(0u)[0].width; }
int SuperPointSingleImp::getInputHeight() const { return task_->getInputTensor(0u)[0].height; }

// Multi-threaded implementation
SuperPointMultiImp::SuperPointMultiImp(const std::string& model_name, int num_threads)
    : SuperPoint(model_name), num_threads_(num_threads) {
  // Always create exactly 4 DPU runners regardless of input parameter
  for (int i = 0; i < NUM_DPU_RUNNERS; ++i) {
    runners_.emplace_back(vitis::ai::DpuTask::create(model_name));
  }
  input_tensors_ = runners_[0]->getInputTensor(0u);
  sWidth = input_tensors_[0].width;
  sHeight = input_tensors_[0].height;
  batch_ = input_tensors_[0].batch;
  chans_ = {65,256};
  auto output_tensors = sort_tensors(runners_[0]->getOutputTensor(0u), chans_);
  channel1 = output_tensors[0].channel;
  channel2 = output_tensors[1].channel;
  outputH = output_tensors[0].height;
  outputW = output_tensors[0].width;
  output2H = output_tensors[1].height;
  output2W = output_tensors[1].width;
  conf_thresh = 0.015;

  LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
    << "tensor1 info : " << output_tensors[0].height << " " << output_tensors[0].width  << " " << output_tensors[0].channel << endl
    << "tensor2 info : " << output_tensors[1].height << " " << output_tensors[1].width  << " " << output_tensors[1].channel << endl;

  outputSize1 = output_tensors[0].channel * output_tensors[0].height * output_tensors[0].width;
  outputSize2 = output_tensors[1].channel * output_tensors[1].height * output_tensors[1].width;
}

SuperPointMultiImp::~SuperPointMultiImp() {}

size_t SuperPointMultiImp::get_input_batch() { return runners_[0]->get_input_batch(0, 0); }
int SuperPointMultiImp::getInputWidth() const {
  return runners_[0]->getInputTensor(0u)[0].width;
}
int SuperPointMultiImp::getInputHeight() const {
  return runners_[0]->getInputTensor(0u)[0].height;
}

// Pre-processing thread function
DpuInferenceTask SuperPointMultiImp::pre_process(const PreProcessTask& input_image) {
  __TIC__(PREPROCESS)
  // scale0 is the scale factor for the input tensor
  float scale0 = vitis::ai::library::tensor_scale(input_tensors_[0]);
  
  float mean = 0;
  float total_scale = scale0 * 0.00392157f;  // 1/255.0
  
  // Pre-process images in the assigned range
    const cv::Mat& img = input_image.image;
    DpuInferenceTask task;
    task.index = input_image.index;

    // Resize image if needed
    __TIC__(RESIZE)
    cv::Mat resized_img;
    if (img.rows == sHeight && img.cols == sWidth) {
      resized_img = img;
    } else {
      cv::resize(img, resized_img, cv::Size(sWidth, sHeight));
    }
    __TOC__(RESIZE)
    
    task.scale_w = img.cols / static_cast<float>(sWidth);
    task.scale_h = img.rows / static_cast<float>(sHeight);

    // Convert to grayscale
    __TIC__(SET_IMG)
    cv::Mat gray_img;
    if (img.channels() == 3) {
      cv::cvtColor(resized_img, gray_img, cv::COLOR_BGR2GRAY);
    } else {
      gray_img = resized_img;
    }

    // Allocate memory for input data
    task.input_data.resize(sWidth * sHeight);
    
    // Optimize conversion to int8_t with scale
#ifdef ENABLE_NEON
    // NEON optimization would go here
    // But keeping scalar implementation for clarity
#endif
    for (int j = 0; j < gray_img.rows * gray_img.cols; ++j) {
      task.input_data[j] = static_cast<int8_t>((gray_img.data[j] - mean) * total_scale);
    }
  __TOC__(SET_IMG)
  __TOC__(PREPROCESS)
  // Enqueue task
  return task;
}
// DPU inference thread function
 DpuInferenceResult SuperPointMultiImp::dpu_inference(const DpuInferenceTask& task, size_t runnerIdx) {
  std::unique_ptr<vitis::ai::DpuTask> runner = std::move(runners_[runnerIdx]);
  __TIC__(DPU_INFERENCE_TOTAL)
  // Prepare input tensor
  auto input_tensors = runner->getInputTensor(0u);
  int8_t* input_data = (int8_t*)input_tensors[0].get_data(0);
  
  // Copy input data efficiently
  __TIC__(MEMCOPY_INPUT)
  std::memcpy(input_data, task.input_data.data(), task.input_data.size());
  __TOC__(MEMCOPY_INPUT)

  // Run DPU inference
  __TIC__(DPU_RUN)
  runner->run(0u);
  __TOC__(DPU_RUN)

  // Get output tensors
  auto output_tensors = sort_tensors(runner->getOutputTensor(0u), chans_);

  // Prepare result
  DpuInferenceResult result;
  result.index = task.index;
  result.scale_w = task.scale_w;
  result.scale_h = task.scale_h;

  // Copy output tensors efficiently
  __TIC__(MEMCOPY_OUTPUT)
  int8_t* out1 = (int8_t*)output_tensors[0].get_data(0);
  int8_t* out2 = (int8_t*)output_tensors[1].get_data(0);

  size_t size1 = output_tensors[0].size / output_tensors[0].batch;
  size_t size2 = output_tensors[1].size / output_tensors[1].batch;

  result.output_data1.resize(size1);
  result.output_data2.resize(size2);
  
  std::memcpy(result.output_data1.data(), out1, size1);
  std::memcpy(result.output_data2.data(), out2, size2);
  __TOC__(MEMCOPY_OUTPUT)

  // Get output scales
  result.scale1 = vitis::ai::library::tensor_scale(output_tensors[0]);
  result.scale2 = vitis::ai::library::tensor_scale(output_tensors[1]);

  __TOC__(DPU_INFERENCE_TOTAL)
  runners_[runnerIdx] = std::move(runner);  // Reassign the runner back to the pool

  return result;
}

// Function to post process a single result
SuperPointResult SuperPointMultiImp::post_process(const DpuInferenceResult& result) {
  SuperPointResult sp_result;
  sp_result.index = result.index;
  sp_result.scale_w = result.scale_w;
  sp_result.scale_h = result.scale_h;

  // Post-processing steps
  const int8_t* out1 = result.output_data1.data();
  const int8_t* out2 = result.output_data2.data();

  float scale1 = result.scale1;
  float scale2 = result.scale2;

  vector<float> output1(outputSize1);

  // Softmax
  __TIC__(SOFTMAX)
#ifndef HW_SOFTMAX
  for (int i = 0; i < outputH * outputW; ++i) {
    float sum{0.0f};
    int pos = i * channel1;
    for (int j = 0; j < channel1; ++j) {
      output1[pos + j] = std::exp(out1[j + pos] * scale1);
      sum += output1[pos + j];
    }
    for (int j = 0; j < channel1; ++j) {
      output1[pos + j] /= sum;
    }
  }
#else
  vitis::ai::softmax(out1, scale1, channel1, outputH * outputW, output1.data());
#endif
  __TOC__(SOFTMAX)

  // Heatmap processing
  __TIC__(HEATMAP)
  int reduced_size = (channel1 - 1) * outputH * outputW;
  vector<float> heatmap(reduced_size);
  // Remove heatmap[-1,:,:]
  for (size_t i = 0; i < outputH * outputW; i++) {
    memcpy(heatmap.data() + i * (channel1 - 1), output1.data() + i * channel1,
           sizeof(float) * (channel1 - 1));
  }
  __TOC__(HEATMAP)

  // Keypoint detection
  __TIC__(SORT)
  vector<float> tmp;
  tmp.reserve(reduced_size);
  vector<int> xs, ys;
  vector<size_t> keep_inds;
  vector<float> ptscore;
  for (size_t m = 0u; m < outputH; ++m) {
    for (size_t i = 0u; i < 8; ++i) {
      for (size_t n = 0u; n < outputW; ++n) {
        for (size_t j = 0u; j < 8; ++j) {
          tmp.push_back(heatmap.at(i * 8 + j + (m * outputW + n) * 64));  // transpose heatmap
          if (tmp.back() > conf_thresh) {
            ys.push_back(m * 8 + i);
            xs.push_back(n * 8 + j);
            ptscore.push_back(tmp.back());
          }
        }
      }
    }
  }
  __TOC__(SORT)

  // NMS - using our optimized version
  __TIC__(NMS)
  nms_fast(xs, ys, ptscore, keep_inds, sWidth, sHeight);
  __TOC__(NMS)

  // L2 Normalization - using our optimized version
  __TIC__(L2_NORMAL)
  vector<float> output2(outputSize2);
  L2_normalization(out2, scale2, channel2, output2H * output2W, output2.data());
  __TOC__(L2_NORMAL)

  // Extract keypoints
  __TIC__(DESC)
  for (size_t i = 0; i < keep_inds.size(); ++i) {
    std::pair<float, float> pt;
    pt.first = float(xs[keep_inds[i]]);
    pt.second = float(ys[keep_inds[i]]);
    sp_result.keypoints.push_back(pt);
  }

  // Descriptor extraction - use optimized bilinear sampling
  bilinear_sample(output2.data(), output2H, output2W, channel2, sp_result.keypoints, sp_result.descriptor);
  __TOC__(DESC)

  return sp_result;
}

std::vector<SuperPointResult>  SuperPointMultiImp::run(const std::vector<cv::Mat>& imgs){
  ThreadSafeQueue<cv::Mat> imageQueue;
  ThreadSafeQueue<SuperPointResult> result_queue;
  bool stop = false;

  // Start pipeline processing in a separate thread
  std::thread pipeline_thread([this, &imageQueue, &result_queue, &stop]() {
    this->run(imageQueue, result_queue, stop);
  });

  // Queue all images
  for (const auto& img : imgs) {
    imageQueue.enqueue(img);
  }
  size_t img_count = imgs.size();
  // Signal that no more images will be added
  stop = true;
  
  // Collect results
  std::vector<SuperPointResult> results;
  SuperPointResult result;

  size_t count = 0;
  while (count < img_count) {
    result_queue.dequeue(result);
    std::cout << "Result index: " << result.index << std::endl;
    results.push_back(result);
    count++;
  }
  
  // Wait for pipeline to finish
  if (pipeline_thread.joinable()) {
    pipeline_thread.join();
  }
  
  return results;
}

  void SuperPointMultiImp::run(ThreadSafeQueue<cv::Mat>& imageQueue, ThreadSafeQueue<SuperPointResult>& result_queue, const bool& stop) {
    // Create ThreadPools for preprocessing and postprocessing
    ThreadPool preproc_pool(num_threads_);
    ThreadPool postproc_pool(num_threads_);
    ThreadPool dpu_pool(NUM_DPU_RUNNERS);
    
    std::vector<std::future<DpuInferenceTask>> preproc_futures;
    std::vector<std::future<std::pair<DpuInferenceResult,size_t>>> dpu_futures;
    std::vector<std::future<SuperPointResult>> postproc_futures;

    // Keep track of available DPU runners
    std::queue<size_t> available_runners;
    for (size_t i = 0; i < runners_.size(); ++i) {
      available_runners.push(i);
    }
    
    int img_idx = 0; // Number of images processed
    int running_count = 0; // Number of tasks currently running
    
    // Store preprocessed tasks waiting for DPU runners
    std::queue<DpuInferenceTask> preprocessed_tasks;
    
    while (!stop || running_count > 0 || !imageQueue.empty()) {
      // Process new images if runners are available
      cv::Mat img;
      if (!imageQueue.empty() && running_count < NUM_DPU_RUNNERS + 2) {
        while( !imageQueue.empty() && running_count < NUM_DPU_RUNNERS + 2){
          if (!imageQueue.dequeue(img)) break;
          img_idx++;
          preproc_futures.push_back(
            preproc_pool.enqueue([this, img, img_idx]() {
              PreProcessTask task;
              task.index = img_idx;
              task.image = img;
              return this->pre_process(task);
            })
          );
          LOG_IF(INFO, ENV_PARAM(DEBUG_THREADS)) << "Enqueued image for preprocessing: " << img_idx;        
          running_count++;
        }
      }

      // First, check if we have any preprocessed tasks waiting that can now be processed
      while (!preprocessed_tasks.empty() && !available_runners.empty()) {
        auto task = preprocessed_tasks.front();
        preprocessed_tasks.pop();
        
        size_t runner_idx = available_runners.front();
        available_runners.pop();
        
        dpu_futures.push_back(
          dpu_pool.enqueue([this, task, runner_idx]() {
            auto result = this->dpu_inference(task, runner_idx);
            return std::make_pair(result, runner_idx);
          })
        );
        LOG_IF(INFO, ENV_PARAM(DEBUG_THREADS)) << "Enqueued image for DPU inference: " << task.index;
      }

      // Process completed preprocessing tasks
      auto preproc_it = preproc_futures.begin();
      while (preproc_it != preproc_futures.end()) {
        // Only check if the future is ready without blocking
        if (preproc_it->wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
          // Get the task from the future - this consumes the future state
          auto task = preproc_it->get();
          
          // Either send directly to DPU processing or store for later
          if (!available_runners.empty()) {
            size_t runner_idx = available_runners.front();
            available_runners.pop();
            
            dpu_futures.push_back(
              dpu_pool.enqueue([this, task, runner_idx]() {
                auto result = this->dpu_inference(task, runner_idx);
                return std::make_pair(result, runner_idx);
              })
            );
            LOG_IF(INFO, ENV_PARAM(DEBUG_THREADS)) << "Enqueued image for DPU inference: " << task.index;
          } else {
            // No runners available, queue for later
            preprocessed_tasks.push(task);
          }
          
          // Remove the processed future
          preproc_it = preproc_futures.erase(preproc_it);
          continue;  // Skip the increment since we already moved to the next item
        } else {
          ++preproc_it;
        }
      }

      // Process completed DPU inference tasks
      auto dpu_it = dpu_futures.begin();
      while (dpu_it != dpu_futures.end()) {
        if (dpu_it->wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
          try {
            auto pair_result = dpu_it->get();
            auto result = pair_result.first;
            auto runner_idx = pair_result.second;
            
            // Return the runner to the available pool
            available_runners.push(runner_idx);
            
            postproc_futures.push_back(
              postproc_pool.enqueue([this, result]() {
                return this->post_process(result);
              })
            );
            LOG_IF(INFO, ENV_PARAM(DEBUG_THREADS)) << "Enqueued image for postprocessing: " << result.index;
          } catch (const std::exception& e) {
            LOG(ERROR) << "Exception in DPU inference: " << e.what();
          }
          
          // Remove the processed future
          dpu_it = dpu_futures.erase(dpu_it);
          continue;  // Skip the increment
        } else {
          ++dpu_it;
        }
      }

      // Process completed postprocessing tasks
      auto postproc_it = postproc_futures.begin();
      while (postproc_it != postproc_futures.end()) {
        if (postproc_it->wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
          try {
            auto result = postproc_it->get();
            // Queue the result
            result_queue.enqueue(result);
            running_count--;
            LOG_IF(INFO, ENV_PARAM(DEBUG_THREADS)) << "Enqueued result for output: " << result.index;
          } catch (const std::exception& e) {
            LOG(ERROR) << "Exception in postprocessing: " << e.what();
            running_count--;
          }
          
          // Remove the processed future
          postproc_it = postproc_futures.erase(postproc_it);
          continue;  // Skip the increment
        } else {
          ++postproc_it;
        }
      }
      
      // Add a small sleep to prevent busy waiting if no tasks are available
      if (preproc_futures.empty() && dpu_futures.empty() && postproc_futures.empty() && imageQueue.empty() && !stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    
    if(!preproc_futures.empty() || !dpu_futures.empty() || !postproc_futures.empty() || !imageQueue.empty() || preprocessed_tasks.size() >0){
       LOG_IF(INFO, ENV_PARAM(DEBUG_THREADS)) << "Some tasks are still running or waiting.";
    }


    /*// Wait for all remaining futures to complete
    for (auto& f : preproc_futures) {
      if (f.valid()) {
        try {
          auto task = f.get();
          SuperPointResult dummy;
          dummy.index = task.index;
          result_queue.enqueue(dummy);
        } catch (const std::exception& e) {
          LOG(ERROR) << "Error waiting for preprocessing tasks: " << e.what();
        }
      }
    }
    
    for (auto& f : dpu_futures) {
      if (f.valid()) {
        try {
          f.wait();
        } catch (const std::exception& e) {
          LOG(ERROR) << "Error waiting for DPU tasks: " << e.what();
        }
      }
    }
    
    for (auto& f : postproc_futures) {
      if (f.valid()) {
        try {
          auto result = f.get();
          result_queue.enqueue(result);
        } catch (const std::exception& e) {
          LOG(ERROR) << "Error waiting for postprocessing tasks: " << e.what();
        }
      }
    }
    */
  }

}  // namespace ai
}  // namespace vitis
