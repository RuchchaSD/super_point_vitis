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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "./superpoint.hpp"

using namespace std;
using namespace cv;

void print_usage(char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options] model_name image" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -s            Use single-threaded implementation (default: multi-threaded)" << std::endl;
  std::cout << "  -t <num>      Number of DPU runners to use (default: 2)" << std::endl;
  std::cout << "  -i <num>      Number of iterations (default: 1)" << std::endl;
  std::cout << "Example: " << prog_name << " -s superpoint_tf.xmodel test.jpg" << std::endl;
  std::cout << "         " << prog_name << " -t 4 superpoint_tf.xmodel test.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  // Default parameters
  bool use_single_threaded = false;
  int num_runners = 2;
  int num_iterations = 10;  // Default changed to 10 for throughput calculation
  std::string model_name;
  std::string image_path;
  
  // Parse command line arguments
  int arg_index = 1;
  while (arg_index < argc) {
    std::string arg = argv[arg_index];
    if (arg == "-s") {
      use_single_threaded = true;
      arg_index++;
    } else if (arg == "-t") {
      if (arg_index + 1 < argc) {
        num_runners = std::stoi(argv[arg_index + 1]);
        arg_index += 2;
      } else {
        std::cerr << "Error: Missing value for -t option" << std::endl;
        print_usage(argv[0]);
        return 1;
      }
    } else if (arg == "-i") {
      if (arg_index + 1 < argc) {
        num_iterations = std::stoi(argv[arg_index + 1]);
        arg_index += 2;
      } else {
        std::cerr << "Error: Missing value for -i option" << std::endl;
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
  if (arg_index + 1 < argc) {
    model_name = argv[arg_index];
    image_path = argv[arg_index + 1];
  } else {
    // Use default values if not provided
    std::cout << "Using default model and image" << std::endl;
    model_name = "../superpoint_tf.xmodel";
    image_path = "../test.jpg";
  }
  
  std::cout << "Configuration:" << std::endl;
  std::cout << "- Implementation: " << (use_single_threaded ? "Single-threaded" : "Multi-threaded") << std::endl;
  if (!use_single_threaded) {
    std::cout << "- Number of runners: " << num_runners << std::endl;
  }
  std::cout << "- Model: " << model_name << std::endl;
  std::cout << "- Image: " << image_path << std::endl;
  std::cout << "- Iterations: " << num_iterations << std::endl;
  
  // Read input image
  Mat img = imread(image_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Error: Could not read image: " << image_path << std::endl;
    return 1;
  }
  
  try {
    // Create SuperPoint instance with appropriate implementation
    auto impl_type = use_single_threaded ? 
                    vitis::ai::SuperPoint::ImplType::SINGLE_THREADED : 
                    vitis::ai::SuperPoint::ImplType::MULTI_THREADED;
    
    auto superpoint = vitis::ai::SuperPoint::create(model_name, impl_type, num_runners);
    if (!superpoint) {
       std::cerr << "Error: Failed to create SuperPoint instance" << std::endl;
       return 1;
    }

    // Prepare input images
    vector<Mat> imgs;
    for (size_t i = 0; i < superpoint->get_input_batch(); ++i) {
      imgs.push_back(img);
    }
    
    // Run inference
    auto start = chrono::high_resolution_clock::now();
    auto result = superpoint->run(imgs);
    for (int i = 1; i < num_iterations; ++i) {
      result = superpoint->run(imgs);
    }
    auto end = chrono::high_resolution_clock::now();

    // Report timing and throughput
    auto duration = chrono::duration_cast<chrono::milliseconds>((end - start));
    int total_images = superpoint->get_input_batch() * num_iterations;
    float throughput = 1000.0f * total_images / duration.count(); // images per second
    
    std::cout << "Processed " << total_images << " images in " 
              << duration.count() << " ms" << std::endl;
    std::cout << "Average time per batch: " << duration.count() / num_iterations << " ms" << std::endl;
    std::cout << "Average time per image: " << duration.count() / total_images << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " images/second" << std::endl;
    
    // Draw and save results (only for the last result to avoid too many output images)
    for (size_t i = 0; i < superpoint->get_input_batch(); ++i) {
      std::cout << "Image " << i << " has " << result[i].keypoints.size() << " keypoints" << std::endl;
      Mat result_img = imgs[i].clone();
      
      // Draw keypoints
      for (size_t k = 0; k < result[i].keypoints.size(); ++k) {
        circle(result_img, 
               Point(result[i].keypoints[k].first * result[i].scale_w,
                     result[i].keypoints[k].second * result[i].scale_h), 
               2, Scalar(0, 0, 255), -1);
      }
      
      std::string output_filename = "result_superpoint_" + 
                                    std::string(use_single_threaded ? "single_" : "multi_") + 
                                    std::to_string(i) + ".jpg";
      imwrite(output_filename, result_img);
      std::cout << "Saved result to " << output_filename << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Done!" << std::endl;
  return 0;
}
