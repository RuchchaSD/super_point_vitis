// SuperPointFast.h
#pragma once
// #include "SuperPoint.hpp"

#include <string>
#include <vector>

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
#include <chrono>
#include <stdexcept>
#include <cstddef>

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
            std::vector<std::pair<float, float>> keypoints;
            std::vector<std::vector<float>> descriptor;
            float scale_w;
            float scale_h;
        };
        
        
        
        // Common utility functions for both implementations
          inline void L2_normalization(const int8_t* input, float scale, int channel, int group, float* output) {
          #ifdef __ARM_NEON
            // NEON vectorized L2 normalization with fixed indices
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
              
              // Extract the sum using fixed lane indices
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
          
          inline vector<vector<float>> grid_sample(const float* desc_map, const vector<pair<float, float>>& coarse_pts,
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
        


          inline void nms_mask(vector<vector<int>>& grid, int x, int y, int dist_thresh) {
            int h = grid.size();
            int w = grid[0].size();
            for (int i = max(0, x - dist_thresh); i < min(h, x + dist_thresh + 1); ++i) {
              for (int j = max(0, y - dist_thresh); j < min(w, y + dist_thresh + 1); ++j) {
                grid[i][j] = -1;
              }
            }
            grid[x][y] = 1;
          }
          
          inline void nms_old(const vector<int>& xs, const vector<int>& ys, const vector<float>& ptscore,
                        vector<size_t>& keep_inds, const int inputW, const int inputH) {
            vector<vector<int>> grid(inputW, vector<int>(inputH, 0));
            vector<pair<float, size_t>> order;
            int dist_thresh = 4; // Higher means more aggressive NMS
            for (size_t i = 0; i < ptscore.size(); ++i) {
              order.push_back({ptscore[i], i});
            }
            std::stable_sort(order.begin(), order.end(),
                             [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
                               return ls.first > rs.first;
                             });
            vector<size_t> ordered;
            transform(order.begin(), order.end(), back_inserter(ordered),
                      [](auto& km) { return km.second; });
          
            for (size_t _i = 0; _i < ordered.size(); ++_i) {
              size_t i = ordered[_i];
              int x = xs[i];
              int y = ys[i];
              if (grid[x][y] == 0 && x >= dist_thresh && x < inputW - dist_thresh && y >= dist_thresh &&
                  y < inputH - dist_thresh) {
                keep_inds.push_back(i);
                nms_mask(grid, x, y, dist_thresh);
              }
            }
          }







          template<typename T>
          class ThreadSafeQueue {
          public:
              explicit ThreadSafeQueue(std::size_t max_size = 20)
                  : buffer_(max_size),
                    max_size_(max_size),
                    head_(0), tail_(0), count_(0),
                    shutdown_(false)
              {}
          
              // non-copyable
              ThreadSafeQueue(const ThreadSafeQueue&) = delete;
              ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
          
              // --- Producer API ---
              void enqueue(const T& item)            { emplace(item); }
              void enqueue(T&& item)                 { emplace(std::move(item)); }
          
              template<typename Rep, typename Period>
              bool try_enqueue_for(const T& item,
                                   const std::chrono::duration<Rep,Period>& dur)
              {
                  return try_emplace_for(dur, item);
              }
          
              template<typename Rep, typename Period>
              bool try_enqueue_for(T&& item,
                                   const std::chrono::duration<Rep,Period>& dur)
              {
                  return try_emplace_for(dur, std::move(item));
              }
          
              // --- Consumer API ---
              bool dequeue(T& out) {
                  std::unique_lock<std::mutex> lock(mutex_);
                  cond_not_empty_.wait(lock, [this]() {
                      return count_ > 0 || shutdown_;
                  });
                  // if shutdown & empty, signal “no more data”
                  if (shutdown_ && count_ == 0)
                      return false;
          
                  out = std::move(buffer_[head_]);
                  head_ = (head_ + 1) % max_size_;
                  --count_;
                  lock.unlock();
                  cond_not_full_.notify_one();
                  return true;
              }
          
              template<typename Rep, typename Period>
              bool try_dequeue_for(T& out,
                                   const std::chrono::duration<Rep,Period>& dur)
              {
                  std::unique_lock<std::mutex> lock(mutex_);
                  if (!cond_not_empty_.wait_for(lock, dur, [this]() {
                          return count_ > 0 || shutdown_;
                      }))
                      return false;
          
                  if (shutdown_ && count_ == 0)
                      return false;
          
                  out = std::move(buffer_[head_]);
                  head_ = (head_ + 1) % max_size_;
                  --count_;
                  lock.unlock();
                  cond_not_full_.notify_one();
                  return true;
              }
          
              // --- Shutdown & Introspection ---
              void shutdown() {
                  std::lock_guard<std::mutex> lock(mutex_);
                  shutdown_ = true;
                  cond_not_empty_.notify_all();
                  cond_not_full_.notify_all();
              }
          
              bool is_shutdown() const {
                  std::lock_guard<std::mutex> lock(mutex_);
                  return shutdown_;
              }
          
              int size() const {
                  std::lock_guard<std::mutex> lock(mutex_);
                  return static_cast<int>(count_);
              }
          
          private:
              // exactly one mutex + two condvars
              mutable std::mutex              mutex_;
              std::condition_variable         cond_not_empty_;
              std::condition_variable         cond_not_full_;
          
              // ring buffer state
              std::vector<T>                  buffer_;
              const std::size_t               max_size_;
              std::size_t                     head_, tail_, count_;
          
              bool                            shutdown_;
          
              // helper to emplace a new item (blocking)
              template<typename U>
              void emplace(U&& item) {
                  std::unique_lock<std::mutex> lock(mutex_);
                  cond_not_full_.wait(lock, [this]() {
                      return count_ < max_size_ || shutdown_;
                  });
                  if (shutdown_)
                      throw std::runtime_error("Queue is shutdown");
          
                  buffer_[tail_] = std::forward<U>(item);
                  tail_ = (tail_ + 1) % max_size_;
                  ++count_;
          
                  lock.unlock();
                  cond_not_empty_.notify_one();
              }
          
              // helper to emplace with timeout
              template<typename Dur, typename U>
              bool try_emplace_for(const Dur& dur, U&& item) {
                  std::unique_lock<std::mutex> lock(mutex_);
                  if (!cond_not_full_.wait_for(lock, dur, [this]() {
                          return count_ < max_size_ || shutdown_;
                      }))
                      return false;
                  if (shutdown_)
                      return false;
          
                  buffer_[tail_] = std::forward<U>(item);
                  tail_ = (tail_ + 1) % max_size_;
                  ++count_;
          
                  lock.unlock();
                  cond_not_empty_.notify_one();
                  return true;
              }
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


        // Multi-threaded implementation
        class SuperPointFast {
            public:
            SuperPointFast(const std::string& model_name, int num_threads);
        
            public:
            virtual ~SuperPointFast();
            virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs);

            void run(ThreadSafeQueue<InputQueueItem>& input_queue, ThreadSafeQueue<SuperPointResult>& output_queue);

            virtual size_t get_input_batch() ;
            virtual int getInputWidth() const ;
            virtual int getInputHeight() const ;
        
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
        
            SuperPointResult process_result(const DpuInferenceResult& result);
            
        
            private:
            static const int NUM_DPU_RUNNERS = 4;  // Fixed number of DPU runners
            int num_threads_;  // Number of pre/post-processing threads
            std::mutex results_mutex_;  // Mutex for synchronizing results access
            // SuperPointFast.h  (inside class SuperPointFast, private:)
            std::thread pipeline_thread_;

        
            std::vector<std::unique_ptr<vitis::ai::DpuTask>> runners_;
            std::vector<SuperPointResult> results_;
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
            float conf_thresh;
            size_t outputSize1;
            size_t outputSize2;
        };
    
    }
}