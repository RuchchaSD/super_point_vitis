//SuperPointSingleImp.h
#pragma once
#include "SuperPoint.hpp"

namespace vitis {
    namespace ai {
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
    }
}