#include "SuperPointSingleImp.h"

namespace vitis {
namespace ai {

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
  
  SuperPointResult SuperPointSingleImp::run(const cv::Mat& img){
    throw std::runtime_error("Single-threaded implementation does not support single image input");
  }
  
  size_t SuperPointSingleImp::get_input_batch() { return task_->get_input_batch(0, 0); }
  int SuperPointSingleImp::getInputWidth() const { return task_->getInputTensor(0u)[0].width; }
  int SuperPointSingleImp::getInputHeight() const { return task_->getInputTensor(0u)[0].height; }

} // namespace ai
} // namespace vitis
