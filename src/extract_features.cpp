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
#include <glog/logging.h>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "SuperPointFast.h"
#include "FeatureIO.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

// Convert SuperPoint keypoints to OpenCV KeyPoints format
std::vector<cv::KeyPoint> convertToKeypoints(const vitis::ai::SuperPointResult& result) {
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(result.keypoints.size());
    
    for (size_t i = 0; i < result.keypoints.size(); ++i) {
        const auto& kp = result.keypoints[i];
        // Scale keypoint coordinates to original image size
        float x = kp.first * result.scale_w;
        float y = kp.second * result.scale_h;
        
        // Create OpenCV keypoint (with reasonable defaults for other parameters)
        cv::KeyPoint cvKeypoint(x, y, 8.0f);  // 8.0 is a common default size
        cvKeypoint.response = 1.0f; // Default response        
        cvKeypoint.octave = 0;  // Default octave
        cvKeypoint.class_id = -1;  // No class ID
        
        keypoints.push_back(cvKeypoint);
    }
    
    return keypoints;
}

// Convert SuperPoint descriptors to OpenCV Mat format
cv::Mat convertToDescriptors(const vitis::ai::SuperPointResult& result) {
    // Get number of keypoints and descriptor dimension
    int numKeypoints = result.keypoints.size();
    int descDim = 256; // SuperPoint uses 256-dim descriptors
    
    // Create a matrix with one row per keypoint and 256 columns (SuperPoint descriptor size)
    cv::Mat descriptors(numKeypoints, descDim, CV_32F);
    
    // Fill descriptor matrix with values
    for (int i = 0; i < numKeypoints; i++) {
        if (i < result.descriptor.size()) {
            const auto& desc = result.descriptor[i];
            // Copy descriptor values to matrix row
            for (int j = 0; j < std::min(descDim, (int)desc.size()); j++) {
                descriptors.at<float>(i, j) = desc[j];
            }
        }
    }
    
    return descriptors;
}

void print_usage(char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options] model_name input_folder output_folder" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -t <num>      Number of preprocessing/postprocessing threads to use (default: 2)" << std::endl;
    std::cout << "                Note: DPU runners fixed at 4" << std::endl;
    std::cout << "  -f <ext>      File extension filter (default: jpg)" << std::endl;
    std::cout << "Example: " << prog_name << " -t 4 superpoint_tf.xmodel ./imgs ./features" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    int num_threads = 2;
    std::string file_ext = "jpg";
    std::string model_name;
    std::string input_folder;
    std::string output_folder;
    
    // Parse command line arguments
    int arg_index = 1;
    while (arg_index < argc) {
        std::string arg = argv[arg_index];
        if (arg == "-t") {
            if (arg_index + 1 < argc) {
                num_threads = std::stoi(argv[arg_index + 1]);
                arg_index += 2;
            } else {
                std::cerr << "Error: Missing value for -t option" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-f") {
            if (arg_index + 1 < argc) {
                file_ext = argv[arg_index + 1];
                arg_index += 2;
            } else {
                std::cerr << "Error: Missing value for -f option" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            break;
        }
    }
    
    // Get positional arguments
    if (arg_index + 2 < argc) {
        model_name = argv[arg_index];
        input_folder = argv[arg_index + 1];
        output_folder = argv[arg_index + 2];
    } else {
        std::cerr << "Error: Missing required positional arguments" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Create output directory structure
    std::string kpts_folder = output_folder + "/kpts";
    std::string desc_folder = output_folder + "/desc";
    
    try {
        if (!fs::exists(output_folder)) {
            fs::create_directories(output_folder);
        }
        if (!fs::exists(kpts_folder)) {
            fs::create_directories(kpts_folder);
        }
        if (!fs::exists(desc_folder)) {
            fs::create_directories(desc_folder);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directories: " << e.what() << std::endl;
        return 1;
    }
    
    // Get list of image files
    std::vector<fs::path> image_paths;
    try {
        for (const auto& entry : fs::directory_iterator(input_folder)) {
            if (entry.is_regular_file() && entry.path().extension() == "." + file_ext) {
                image_paths.push_back(entry.path());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading input directory: " << e.what() << std::endl;
        return 1;
    }
    
    if (image_paths.empty()) {
        std::cerr << "No image files found in " << input_folder << " with extension ." << file_ext << std::endl;
        return 1;
    }
    
    std::cout << "SuperPoint Feature Extraction:" << std::endl;
    std::cout << "- Number of pre/post-processing threads: " << num_threads << std::endl;
    std::cout << "- Number of DPU runners: 4 (fixed)" << std::endl;
    std::cout << "- Model: " << model_name << std::endl;
    std::cout << "- Input folder: " << input_folder << std::endl;
    std::cout << "- Output folder: " << output_folder << std::endl;
    std::cout << "- Keypoints folder: " << kpts_folder << std::endl;
    std::cout << "- Descriptors folder: " << desc_folder << std::endl;
    std::cout << "- File extension: " << file_ext << std::endl;
    std::cout << "- Total images found: " << image_paths.size() << std::endl;

    try {
        // Initialize SuperPointFast
        auto superpoint = vitis::ai::SuperPointFast(model_name, num_threads);
        
        // Create thread-safe queues
        vitis::ai::ThreadSafeQueue<vitis::ai::InputQueueItem> input_queue;
        vitis::ai::ThreadSafeQueue<vitis::ai::SuperPointResult> output_queue;
        
        // Atomic flag for rate limiting
        std::atomic<bool> hold_images{false};
        
        // Start the SuperPointFast processor 
        superpoint.run(input_queue, output_queue, hold_images);
        
        // Start producer thread to feed images when rate limiting allows
        std::thread producer_thread([&image_paths, &input_queue, &hold_images]() {
            size_t count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (const auto& img_path : image_paths) {
                // Check rate limiting - wait until processing pipeline is ready for more
                size_t held_count = 0;
                while (hold_images.load()) {
                    ++held_count;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (held_count > 0) {
                    std::cout << "Producer waited " << held_count*5 << " ms for back-pressure\n";
                }
                
                // Read image
                cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Warning: Could not read image: " << img_path.string() << std::endl;
                    continue;
                }
                
                // Create input queue item (store filename in the data field)
                vitis::ai::InputQueueItem item;
                item.index = count++;
                item.image = img;
                item.name = img_path.filename().string();
                
                // Add to queue
                input_queue.enqueue(item);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                
                // std::cout << "Enqueued image " << count << ": " << img_path.filename().string() << std::endl;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "All " << count << " images queued in " << duration.count() << " ms" << std::endl;
            
            // Signal that no more images will be added
            input_queue.shutdown();
        });
        
        // Start consumer thread to process results and save features
        std::thread consumer_thread([&output_queue, &kpts_folder, &desc_folder]() {
            size_t result_count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            vitis::ai::SuperPointResult result;
            while (output_queue.dequeue(result)) {
                result_count++;
                
                // Extract base filename without extension
                std::string base_filename = fs::path(result.name).stem().string();
                if (base_filename.empty()) {
                    // If name wasn't passed through, use the index
                    base_filename = "image_" + std::to_string(result.index);
                }
                
                // Convert SuperPoint keypoints/descriptors to OpenCV format
                std::vector<cv::KeyPoint> keypoints = convertToKeypoints(result);
                cv::Mat descriptors = convertToDescriptors(result);
                
                // Define filenames for keypoints and descriptors
                std::string kpts_filename = kpts_folder + "/" + base_filename + ".kp";
                std::string desc_filename = desc_folder + "/" + base_filename + ".desc";
                
                // Save keypoints and descriptors
                bool saved = FeatExtraction::FeatureIO::saveFeatures(keypoints, descriptors, 
                                                                    kpts_filename, desc_filename);
                
                if (saved) {
                    std::cout << "Saved features for " << base_filename 
                              << " (keypoints: " << keypoints.size() << ")" << std::endl;
                } else {
                    std::cerr << "Error saving features for " << base_filename << std::endl;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "All " << result_count << " feature sets extracted and saved in " 
                      << duration.count() << " ms" << std::endl;
        });
        
        // Wait for all threads to complete
        producer_thread.join();
        consumer_thread.join();
        
        // Calculate statistics
        std::cout << "Feature extraction completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}