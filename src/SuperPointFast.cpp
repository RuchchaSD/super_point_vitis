//SuperPointFast.cpp
#include "SuperPointFast.h"
#include <iomanip>


// Implementation of SuperPointFast methods
SuperPointFast::SuperPointFast(const std::string& model_name, int num_threads)
{
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
    // conf_thresh = 0.007; // Helitha
    conf_thresh = 0.015; // Xilinx
    
    scale0 = vitis::ai::library::tensor_scale(input_tensors_[0]);

    LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
    << "tensor1 info : " << output_tensors[0].height << " " << output_tensors[0].width  << " " << output_tensors[0].channel << endl
    << "tensor2 info : " << output_tensors[1].height << " " << output_tensors[1].width  << " " << output_tensors[1].channel << endl;

    outputSize1 = output_tensors[0].channel * output_tensors[0].height * output_tensors[0].width;
    outputSize2 = output_tensors[1].channel * output_tensors[1].height * output_tensors[1].width;
}

SuperPointFast::~SuperPointFast() {
    if (pipeline_thread_.joinable()) pipeline_thread_.join();
}

size_t SuperPointFast::get_input_batch() { return runners_[0]->get_input_batch(0, 0); }
int SuperPointFast::getInputWidth() const {
    return runners_[0]->getInputTensor(0u)[0].width;
}
int SuperPointFast::getInputHeight() const {
    return runners_[0]->getInputTensor(0u)[0].height;
}

// Pre-processing thread function
void SuperPointFast::pre_process(const std::vector<cv::Mat>& input_images,
                            ThreadSafeQueue<DpuInferenceTask>& task_queue,
                            int start_idx, int end_idx) {
    __TIC__(PREPROCESS)
    // Pre-process images in the assigned range
    for (int i = start_idx; i < end_idx; ++i) {
        // Enqueue task
        task_queue.enqueue(
            pre_process_image(input_images[i], i)
        );
    }
    __TOC__(PREPROCESS)
}


DpuInferenceTask SuperPointFast::pre_process_image(const cv::Mat& img, int idx) 
{
    float mean = 0;
    float total_scale = scale0 * 1/255.0;  // 1/255.0

    DpuInferenceTask task;
    task.index = idx;
    task.img = img;

    // Resize image if needed
    cv::Mat resized_img;
    if (img.rows == sHeight && img.cols == sWidth) {
        resized_img = img;
    } else {
        cv::resize(img, resized_img, cv::Size(sWidth, sHeight));
    }

    task.scale_w = img.cols / static_cast<float>(sWidth);
    task.scale_h = img.rows / static_cast<float>(sHeight);

    // Convert to grayscale
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

    // Enqueue task
    return task;
}


DpuInferenceTask SuperPointFast::pre_process_image(InputQueueItem inputItem) 
{
    float mean = 0;
    float total_scale = scale0 * 1/255.0;  // 1/255.0

    DpuInferenceTask task;
    task.index = inputItem.index;
    task.img = inputItem.image;
    task.name = inputItem.name;

    auto img = inputItem.image;

    // Resize image if needed
    cv::Mat resized_img;
    if (img.rows == sHeight && img.cols == sWidth) {
        resized_img = img;
    } else {
        cv::resize(img, resized_img, cv::Size(sWidth, sHeight));
    }

    task.scale_w = img.cols / static_cast<float>(sWidth);
    task.scale_h = img.rows / static_cast<float>(sHeight);

    // Convert to grayscale
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

    // Enqueue task
    return task;
}

void SuperPointFast::dpu_inference(ThreadSafeQueue<DpuInferenceTask>& task_queue,
                                ThreadSafeQueue<DpuInferenceResult>& result_queue) {
    __TIC__(DPU_INFERENCE_TOTAL)
    // Create thread pool based on available DPU runners
    std::vector<std::thread> worker_threads;

    // Launch one worker thread per DPU runner
    for (size_t i = 0; i < runners_.size(); ++i) {
        worker_threads.emplace_back([this, i, &task_queue, &result_queue]() {
            auto& runner = runners_[i];
            int tasks_processed = 0;
            
            while (true) {
                // Get next task from queue
                DpuInferenceTask task;
                if (!task_queue.dequeue(task)) {
                    break;
                }

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
                result.img = task.img;
                result.name = task.name;
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

                // Enqueue result for post-processing
                result_queue.enqueue(result);
                tasks_processed++;
            }
        });
    }

    // Wait for all worker threads to finish
    for (size_t i = 0; i < worker_threads.size(); ++i) {
        worker_threads[i].join();
    }

    // Signal that all processing is complete
    result_queue.shutdown();
    __TOC__(DPU_INFERENCE_TOTAL)
}

void SuperPointFast::post_process(ThreadSafeQueue<DpuInferenceResult>& result_queue,
                                size_t thread_idx, size_t num_threads) {
    __TIC__(POSTPROCESS_TOTAL)
    int processed_count = 0;

    while (true) {
        DpuInferenceResult result;
        if (!result_queue.dequeue(result)) {
            break; // Queue empty and shutdown
        }

        // Process result
        __TIC__(PROCESS_RESULT)
        SuperPointResult sp_result = process_result(result);
        __TOC__(PROCESS_RESULT)

        // Store result in a thread-safe manner
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_[result.index] = sp_result;
        }

        processed_count++;
    }

    __TOC__(POSTPROCESS_TOTAL)
}

// Function to process a single result
SuperPointResult SuperPointFast::process_result(const DpuInferenceResult& result) {
    SuperPointResult sp_result;
    sp_result.index = result.index;
    sp_result.img = result.img;
    sp_result.name = result.name;
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
    // float max = 0, min = 0;
    tmp.reserve(reduced_size);
    vector<int> xs, ys;
    vector<size_t> keep_inds;
    vector<float> ptscore;
    for (size_t m = 0u; m < outputH; ++m) {
        for (size_t i = 0u; i < 8; ++i) {
            for (size_t n = 0u; n < outputW; ++n) {
                for (size_t j = 0u; j < 8; ++j) {
                    tmp.push_back(heatmap.at(i * 8 + j + (m * outputW + n) * 64));  // transpose heatmap
                    // std::cout << "KP Num: "<< i * 8 + j + (m * outputW + n) * 64 << " | reliability: " << tmp.back() << std::endl;
                    // allScores.push_back(tmp.back());
                    if (tmp.back() > conf_thresh) {
                        ys.push_back(m * 8 + i);
                        xs.push_back(n * 8 + j);
                        ptscore.push_back(tmp.back());
                        // if (tmp.back() > max) {
                        // max = tmp.back();
                        // }
                    }
                }
            }
        }
    }
    __TOC__(SORT)


    // const int num_bins = 10;
    // std::vector<int> histogram(num_bins, 0);
    // float bin_width = (max - min) / num_bins;
    // int total = 0;

    // for (size_t i = 0; i < tmp.size(); ++i) {
    //     if (tmp[i] > min && tmp[i] < max) {
    //         int bin_index = static_cast<int>((tmp[i] - min) / bin_width);
    //         if (bin_index >= 0 && bin_index < num_bins) {
    //             histogram[bin_index]++;
    //             total++;
    //         }
    //     }
    // }
    // std::cout << "Keypoint confidence score distribution:" << std::endl;
    // std::cout << "Total keypoints: " << tmp.size() << ", Valid keypoints: " << ptscore.size() << std::endl;
    // for (int i = 0; i < num_bins; ++i) {
    //     float bin_start = min + i * bin_width;
    //     float bin_end = bin_start + bin_width;
    //     std::cout << "[" << std::fixed << std::setprecision(3) << bin_start << " - " << bin_end << "): " << histogram[i] << std::endl;
    // }

    // NMS - using our optimized version
    __TIC__(NMS)
    // nms_fast(xs, ys, ptscore, keep_inds, sWidth, sHeight);
    nms_old(xs, ys, ptscore, keep_inds, sWidth, sHeight);
    __TOC__(NMS)

    // L2 Normalization - using our optimized version
    __TIC__(L2_NORMAL)
    vector<float> output2(outputSize2);
    L2_normalization(out2, scale2, channel2, output2H * output2W, output2.data());
    __TOC__(L2_NORMAL)

    // Extract keypoints and create OpenCV keypoints vector
    __TIC__(DESC)
    std::vector<std::pair<float, float>> kps; // Temporary for bilinear sampling
    sp_result.keypoints_cv.clear();
    sp_result.keypoints_cv.reserve(keep_inds.size());
    
    // Extract keypoints
    for (size_t i = 0; i < keep_inds.size(); ++i) {
        float x = float(xs[keep_inds[i]]);
        float y = float(ys[keep_inds[i]]);
        
        // Create OpenCV keypoint
        cv::KeyPoint kp(
            x * result.scale_w,  // scaled x
            y * result.scale_h,  // scaled y
            8.0f,                // size
            -1.0f,               // angle
            ptscore[keep_inds[i]], // response
            0,                   // octave
            -1                   // class id
        );
        
        sp_result.keypoints_cv.push_back(kp);
        
        // Keep the unscaled points for descriptor sampling
        kps.push_back(std::make_pair(x, y));
    }
    
    // Descriptor extraction - use optimized bilinear sampling
    std::vector<std::vector<float>> descriptors;
    bilinear_sample(output2.data(), output2H, output2W, channel2, kps, descriptors);
    
    // Convert descriptors to OpenCV Mat format
    if (!descriptors.empty()) {
        int numKeypoints = descriptors.size();
        int descDim = descriptors[0].size();
        
        cv::Mat cv_descriptors(numKeypoints, descDim, CV_32F);
        for (int i = 0; i < numKeypoints; i++) {
            for (int j = 0; j < descDim; j++) {
                cv_descriptors.at<float>(i, j) = descriptors[i][j];
            }
        }
        
        // Assign to result
        sp_result.descriptors_cv = cv_descriptors;
    }
    __TOC__(DESC)

    return sp_result;
}

// Run function with proper promise handling
std::vector<SuperPointResult> SuperPointFast::run(const std::vector<cv::Mat>& imgs) {
    // Create thread-safe queues for the pipeline
    auto task_queue = std::make_shared<ThreadSafeQueue<DpuInferenceTask>>();
    auto result_queue = std::make_shared<ThreadSafeQueue<DpuInferenceResult>>();

    // Resize results vector to match input size
    results_.resize(imgs.size());

    // Create a promise that will be fulfilled when processing is complete
    auto promise = std::make_shared<std::promise<std::vector<SuperPointResult>>>();
    auto future = promise->get_future();

    // Start pre-processing threads (multiple threads based on num_threads_)
    std::vector<std::thread> preproc_threads;
    int images_per_thread = imgs.size() / num_threads_;
    int remaining_images = imgs.size() % num_threads_;

    int start_idx = 0;
    for (int i = 0; i < num_threads_; ++i) {
        int num_images = images_per_thread + (i < remaining_images ? 1 : 0);
        int end_idx = start_idx + num_images;

        if (num_images > 0) {
            preproc_threads.emplace_back([this, &imgs, task_queue, start_idx, end_idx]() {
                this->pre_process(imgs, *task_queue, start_idx, end_idx);
            });
            start_idx = end_idx;
        }
    }

    // Start DPU inference thread (fixed 4 DPU runners internally)
    std::thread dpu_thread([this, task_queue, result_queue]() {
        this->dpu_inference(*task_queue, *result_queue);
    });

    // Start post-processing threads 
    std::vector<std::thread> postproc_threads;
    for (int i = 0; i < num_threads_; ++i) {
        postproc_threads.emplace_back([this, result_queue, i]() {
            this->post_process(*result_queue, i, num_threads_);
        });
    }

    // Wait for all threads to finish
    for (size_t i = 0; i < preproc_threads.size(); ++i) {
        preproc_threads[i].join();
    }

    task_queue->shutdown();

    dpu_thread.join();

    for (size_t i = 0; i < postproc_threads.size(); ++i) {
        postproc_threads[i].join();
    }

    // After all threads have completed, set the promise with the results
    promise->set_value(results_);

    auto result = future.get();
    return result;
}

// Overloaded run function for continuous processing with queues
void SuperPointFast::run(ThreadSafeQueue<InputQueueItem>& input_queue, ThreadSafeQueue<SuperPointResult>& output_queue) 
{
    // If a previous run() is still alive, wait for it.
    if (pipeline_thread_.joinable()) pipeline_thread_.join();

    // Launch the pipeline in its own thread so the caller regains control.
    pipeline_thread_ = std::thread([this, &input_queue, &output_queue]()
    {
        // ───────────────  Stage-local queues  ───────────────
        auto task_queue   = std::make_shared<ThreadSafeQueue<DpuInferenceTask>>(20);
        auto result_queue = std::make_shared<ThreadSafeQueue<DpuInferenceResult>>(50);

        // ─────────────── 1.  Pre-processing stage ───────────────
        std::vector<std::thread> preproc_threads;
        for (int t = 0; t < num_threads_; ++t) 
        {
            preproc_threads.emplace_back([this, &input_queue, task_queue]() 
            {
                while (true) 
                {
                    
                    InputQueueItem item;
                    if (!input_queue.dequeue(item))
                    {         
                        break;
                    }

                    if(input_queue.is_shutdown())
                    {
                        std::cout << "Input queue is shutdowned, preprocessing thread will exit once all images are preprocessed" << std::endl;
                    }

                    task_queue->enqueue(
                        pre_process_image(item)
                    );                            
                }
            }
            );
        }

        // ─────────────── 2.  DPU-inference stage ───────────────
        std::thread dpu_thread([this, task_queue, result_queue]() 
        {
            // Re-use the existing dpu_inference() helper but under our own lifetime.
            this->dpu_inference(*task_queue, *result_queue);
        });

        // ─────────────── 3.  Post-processing stage ───────────────
        std::vector<std::thread> postproc_threads;
        for (int t = 0; t < num_threads_; ++t) {
            postproc_threads.emplace_back([this, result_queue, &output_queue]() {
                DpuInferenceResult raw;
                while (result_queue->dequeue(raw)) {
                    SuperPointResult r = this->process_result(raw);
                    if (!output_queue.is_shutdown()) {
                        output_queue.enqueue(r);
                    }else{
                        throw std::runtime_error("Output queue should not be shutdowned by caller");
                    }

                }
            });
        }

        // ───────────────  Shutdown coordination  ───────────────
        for (auto& th : preproc_threads)  th.join();
        task_queue->shutdown();
        dpu_thread.join();
        result_queue->shutdown();
        for (auto& th : postproc_threads) th.join();
        output_queue.shutdown();               
        
    }
    );
}