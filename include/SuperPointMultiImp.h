// SuperPointMultiImp.h
#pragma once
#include "SuperPoint.hpp"

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

            void shutdown() {
                {
                std::lock_guard<std::mutex> lock(mutex_);
                shutdown_ = true;
                }
                cond_var_.notify_all();
            }
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


        // Multi-threaded implementation
        class SuperPointMultiImp : public SuperPoint {
            public:
            SuperPointMultiImp(const std::string& model_name, int num_threads);
        
            public:
            virtual ~SuperPointMultiImp();
            virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs) override;
            virtual size_t get_input_batch() override;
            virtual int getInputWidth() const override;
            virtual int getInputHeight() const override;
        
            private:
            void pre_process(const std::vector<cv::Mat>& input_images,
                            ThreadSafeQueue<DpuInferenceTask>& task_queue,
                            int start_idx, int end_idx);
            void dpu_inference(ThreadSafeQueue<DpuInferenceTask>& task_queue,
                                ThreadSafeQueue<DpuInferenceResult>& result_queue);
            void post_process(ThreadSafeQueue<DpuInferenceResult>& result_queue,
                            size_t thread_idx, size_t num_threads);
        
            SuperPointResult process_result(const DpuInferenceResult& result);
        
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
    }
}