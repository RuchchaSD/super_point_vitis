//SuperPointFast.cpp
#include "SuperPointFast.h"
#include <iomanip>
#include <xir/graph/subgraph.hpp>
#include <vart/tensor_buffer.hpp>
#include <vart/runner_helper.hpp>
#include <vart/runner.hpp>

// Global atomic variable for confidence threshold
std::atomic<float> g_conf_thresh(0.015f); // Default value = 0.015 (middle of 0.0-0.03 range)

// Helper function to allocate CPU flat tensor buffer
static std::unique_ptr<vart::TensorBuffer> alloc_cpu_flat_tensor_buffer(const xir::Tensor* tensor) {
    return vart::alloc_cpu_flat_tensor_buffer(tensor);
}

// Helper function to get scale factor from tensor
float SuperPointFast::get_tensor_scale(const xir::Tensor* tensor) {
    // Get the scale from tensor attributes
    float scale = 1.0f;
    if (tensor->has_attr("scale")) {
        scale = tensor->get_attr<float>("scale");
    }
    return scale;
}

// Implementation of SuperPointFast methods
SuperPointFast::SuperPointFast(const std::string& model_name, int num_threads)
    : num_threads_(num_threads)
{
    // Load the model graph from xmodel file
    auto graph = xir::Graph::deserialize(model_name);
    auto root_subgraph = graph->get_root_subgraph();
    auto subgraphs = root_subgraph->children_topological_sort();
    
    // Filter for DPU subgraphs only
    auto dpu_subgraphs = std::vector<const xir::Subgraph*>();
    for (auto subgraph : subgraphs) {
        if (subgraph->has_attr("device") &&
            subgraph->get_attr<std::string>("device") == "DPU") {
            dpu_subgraphs.push_back(subgraph);
        }
    }
    
    if (dpu_subgraphs.empty()) {
        throw std::runtime_error("No DPU subgraphs found in the model");
    }

    // Create DPU runner instances
    for (int i = 0; i < NUM_DPU_RUNNERS; ++i) {
        runners_.emplace_back(vart::Runner::create_runner(dpu_subgraphs[0], "run"));
    }
    
    // Get input tensors
    auto input_tensors = runners_[0]->get_input_tensors();
    if (input_tensors.empty()) {
        throw std::runtime_error("No input tensors in model");
    }
    
    input_tensors_ = input_tensors;
    
    // Get input dimensions
    auto in_dims = input_tensors[0]->get_shape();
    batch_ = in_dims[0];
    // NCHW format: in_dims = [batch, channel, height, width]
    sHeight = in_dims[2];
    sWidth = in_dims[3];
    
    // Expected output channels
    chans_ = {65, 256};
    
    // Get output tensors and sort them
    auto output_tensors = sort_tensors(runners_[0]->get_output_tensors(), chans_);
    if (output_tensors.size() < 2) {
        throw std::runtime_error("Expected at least 2 output tensors");
    }
    
    // Get output tensor dimensions
    auto out_dims1 = output_tensors[0]->get_shape();
    auto out_dims2 = output_tensors[1]->get_shape();
    
    channel1 = out_dims1[1]; // NCHW format
    channel2 = out_dims2[1];
    outputH = out_dims1[2];
    outputW = out_dims1[3];
    output2H = out_dims2[2];
    output2W = out_dims2[3];
    
    // Get scale for input tensor
    scale0 = get_tensor_scale(input_tensors[0]);
    
    LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
        << "tensor1 info : " << outputH << " " << outputW << " " << channel1 << endl
        << "tensor2 info : " << output2H << " " << output2W << " " << channel2 << endl;
    
    outputSize1 = channel1 * outputH * outputW;
    outputSize2 = channel2 * output2H * output2W;
}

SuperPointFast::~SuperPointFast() {
    if (pipeline_thread_.joinable()) pipeline_thread_.join();
}

size_t SuperPointFast::get_input_batch() { 
    if (!input_tensors_.empty()) {
        return input_tensors_[0]->get_shape()[0]; 
    }
    return 1; // Default
}

int SuperPointFast::getInputWidth() const {
    if (!input_tensors_.empty()) {
        return input_tensors_[0]->get_shape()[3]; // NCHW format
    }
    return sWidth;
}

int SuperPointFast::getInputHeight() const {
    if (!input_tensors_.empty()) {
        return input_tensors_[0]->get_shape()[2]; // NCHW format
    }
    return sHeight;
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
    task.timestamp = 0.0;  // Default timestamp
    task.filename = "";    // Default empty filename
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
    task.timestamp = inputItem.timestamp;
    task.img = inputItem.image;
    task.filename = inputItem.filename;

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

            auto input_tensors = runner->get_input_tensors();
            auto output_tensors = sort_tensors(runner->get_output_tensors(), chans_);
            
            while (true) {
                // Get next task from queue
                DpuInferenceTask task;
                if (!task_queue.dequeue(task)) {
                    break;
                }
                __TIC__(DPU_INFERENCE_CYCLE)

                // Create input and output buffers using RAII for proper cleanup
                std::vector<std::unique_ptr<vart::TensorBuffer>> owned_input_buffers;
                std::vector<std::unique_ptr<vart::TensorBuffer>> owned_output_buffers;
                std::vector<vart::TensorBuffer*> input_buffers, output_buffers;
                
                try {
                    // Allocate CPU tensor buffers for input
                    auto input_buffer = vart::alloc_cpu_flat_tensor_buffer(input_tensors[0]);
                    input_buffers.push_back(input_buffer.get());
                    owned_input_buffers.push_back(std::move(input_buffer));
                    
                    // Allocate CPU tensor buffers for output
                    for (auto tensor : output_tensors) {
                        auto buffer = vart::alloc_cpu_flat_tensor_buffer(tensor);
                        output_buffers.push_back(buffer.get());
                        owned_output_buffers.push_back(std::move(buffer));
                    }
                    
                    // Get input data buffer
                    auto input_buffer_data = vart::get_tensor_buffer_data(input_buffers[0], 0);
                    int8_t* input_data = static_cast<int8_t*>(input_buffer_data.data);
                    
                    // Copy input data efficiently
                    __TIC__(MEMCOPY_INPUT)
                    std::memcpy(input_data, task.input_data.data(), task.input_data.size());
                    // Sync for write
                    input_buffers[0]->sync_for_write(0, task.input_data.size());
                    __TOC__(MEMCOPY_INPUT)
                    
                    // Run DPU inference (async)
                    __TIC__(DPU_RUN)
                    auto job_id = runner->execute_async(input_buffers, output_buffers);
                    runner->wait(job_id.first, -1);  // Wait for completion (-1 = wait forever)
                    __TOC__(DPU_RUN)
                    
                    // Prepare result
                    DpuInferenceResult result;
                    result.index = task.index;
                    result.timestamp = task.timestamp;
                    result.img = task.img;
                    result.filename = task.filename;
                    result.scale_w = task.scale_w;
                    result.scale_h = task.scale_h;
                    
                    // Copy output tensors efficiently
                    __TIC__(MEMCOPY_OUTPUT)
                    // Sync for read
                    for (size_t j = 0; j < output_buffers.size(); ++j) {
                        output_buffers[j]->sync_for_read(0, output_buffers[j]->get_tensor()->get_data_size());
                    }
                    
                    // Get output tensor data
                    auto output_buffer_data1 = vart::get_tensor_buffer_data(output_buffers[0], 0);
                    auto output_buffer_data2 = vart::get_tensor_buffer_data(output_buffers[1], 0);
                    
                    int8_t* out1 = static_cast<int8_t*>(output_buffer_data1.data);
                    int8_t* out2 = static_cast<int8_t*>(output_buffer_data2.data);
                    
                    size_t size1 = output_buffer_data1.size;
                    size_t size2 = output_buffer_data2.size;
                    
                    result.output_data1.resize(size1);
                    result.output_data2.resize(size2);
                    
                    std::memcpy(result.output_data1.data(), out1, size1);
                    std::memcpy(result.output_data2.data(), out2, size2);
                    __TOC__(MEMCOPY_OUTPUT)
                    
                    // Get output scales
                    result.scale1 = get_tensor_scale(output_tensors[0]);
                    result.scale2 = get_tensor_scale(output_tensors[1]);
                    
                    // Enqueue result for post-processing
                    result_queue.enqueue(result);
                    tasks_processed++;
                } catch (const std::exception& e) {
                    LOG(ERROR) << "Error in DPU inference: " << e.what();
                }
                __TOC__(DPU_INFERENCE_CYCLE)
                
                // Tensor buffers are automatically cleaned up by unique_ptr when they go out of scope
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
        ResultQueueItem sp_result = process_result(result);
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
ResultQueueItem SuperPointFast::process_result(const DpuInferenceResult& result) {
    ResultQueueItem sp_result;
    sp_result.index = result.index;
    sp_result.timestamp = result.timestamp;
    sp_result.image = result.img;
    sp_result.filename = result.filename;

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

     float current_conf_thresh = g_conf_thresh.load(); // Use atomic variable

    if(std::getenv("DUMP_SUPERPOINT_THREADS") != nullptr){
        std::cout << "Confidence threshold: " << current_conf_thresh << std::endl;
    }

    // Keypoint detection
    __TIC__(SORT)
    vector<float> tmp;
    float max = 0, min = 0;
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
                    if (tmp.back() > current_conf_thresh) {
                        ys.push_back(m * 8 + i);
                        xs.push_back(n * 8 + j);
                        ptscore.push_back(tmp.back());
                        if(std::getenv("DUMP_SUPERPOINT_THREADS") != nullptr){
                            if (tmp.back() > max) {
                            max = tmp.back();
                            }
                        }
                    }
                }
            }
        }
    }
    __TOC__(SORT)

    if(std::getenv("DUMP_SUPERPOINT_THREADS") != nullptr){
        max = 0.015;
        const int num_bins = 10;
        std::vector<int> histogram(num_bins, 0);
        float bin_width = (max - min) / num_bins;
        int total = 0;

        for (size_t i = 0; i < tmp.size(); ++i) {
            if (tmp[i] > min && tmp[i] < max) {
                int bin_index = static_cast<int>((tmp[i] - min) / bin_width);
                if (bin_index >= 0 && bin_index < num_bins) {
                    histogram[bin_index]++;
                    total++;
                }
            }
        }
        std::cout << "Keypoint confidence score distribution:" << std::endl;
        std::cout << "Total keypoints: " << tmp.size() << ", Valid keypoints: " << ptscore.size() << std::endl;
        for (int i = 0; i < num_bins; ++i) {
            float bin_start = min + i * bin_width;
            float bin_end = bin_start + bin_width;
            std::cout << "[" << std::fixed << std::setprecision(3) << bin_start << " - " << bin_end << "): " << histogram[i] << std::endl;
        }
    }

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
    sp_result.keypoints.clear();
    sp_result.keypoints.reserve(keep_inds.size());
    
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
        
        sp_result.keypoints.push_back(kp);
        
        // Keep the unscaled points for descriptor sampling
        // kps.push_back(std::make_pair(x, y));
        kps.emplace_back(x, y);
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
        sp_result.descriptors = cv_descriptors;
    }
    __TOC__(DESC)

    return sp_result;
}

// Run function with proper promise handling
std::vector<ResultQueueItem> SuperPointFast::run(const std::vector<cv::Mat>& imgs) {
    // Create thread-safe queues for the pipeline
    auto task_queue = std::make_shared<ThreadSafeQueue<DpuInferenceTask>>();
    auto result_queue = std::make_shared<ThreadSafeQueue<DpuInferenceResult>>();

    // Resize results vector to match input size
    results_.resize(imgs.size());

    // Create a promise that will be fulfilled when processing is complete
    auto promise = std::make_shared<std::promise<std::vector<ResultQueueItem>>>();
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
void SuperPointFast::run(ThreadSafeQueue<InputQueueItem>& input_queue, ThreadSafeQueue<ResultQueueItem>& output_queue) 
{
    // If a previous run() is still alive, wait for it.
    if (pipeline_thread_.joinable()) pipeline_thread_.join();

    // Create a shared_ptr to the Sequencer so it outlives this function
    auto seq = std::make_shared<Sequencer<std::size_t,
              ResultQueueItem,
              ThreadSafeQueue<ResultQueueItem>>>(0, output_queue);

    // Launch the pipeline in its own thread so the caller regains control.
    pipeline_thread_ = std::thread([this, &input_queue, &output_queue, seq]()
    {
        // ───────────────  Stage-local queues  ───────────────
        auto task_queue   = std::make_shared<ThreadSafeQueue<DpuInferenceTask>>(1.25 * NUM_DPU_RUNNERS);
        auto result_queue = std::make_shared<ThreadSafeQueue<DpuInferenceResult>>((int)(1.25 * num_threads_));

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
            this->dpu_inference(*task_queue, *result_queue);
        });

        // ─────────────── 3.  Post-processing stage ───────────────
        std::vector<std::thread> postproc_threads;
        for (int t = 0; t < num_threads_; ++t) {
            postproc_threads.emplace_back([this, result_queue, &seq]() {
                DpuInferenceResult raw;
                while (result_queue->dequeue(raw)) {
                    ResultQueueItem r = this->process_result(raw);
                    seq->push(r.index, std::move(r));
                }
            });
        }

        // ───────────────  Shutdown coordination  ───────────────
        for (auto& th : preproc_threads)  th.join();
        task_queue->shutdown();
        dpu_thread.join();
        result_queue->shutdown();
        for (auto& th : postproc_threads) th.join();
        seq->flush();
        output_queue.shutdown();               
        
    }
    );
}