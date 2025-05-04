#include "SuperPointMultiImp.h"

namespace vitis {
namespace ai {

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

std::vector<SuperPointResult> SuperPointMultiImp::run(const std::vector<cv::Mat>& imgs){
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

// Add the missing method implementation for single image
SuperPointResult SuperPointMultiImp::run(const cv::Mat& img) {
  // Just use the batch version with a single image
  std::vector<cv::Mat> imgs = {img};
  auto results = run(imgs);
  
  // Return the first (and only) result
  if (results.empty()) {
    throw std::runtime_error("Failed to process image in SuperPointMultiImp::run");
  }
  return results[0];
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
}

} // namespace ai
} // namespace vitis