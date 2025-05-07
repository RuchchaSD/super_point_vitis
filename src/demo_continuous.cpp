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
#include "SuperPointFast.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

void print_usage(char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options] model_name input_folder output_folder" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -t <num>      Number of preprocessing/postprocessing threads to use (default: 2)" << std::endl;
  std::cout << "                Note: DPU runners fixed at 4" << std::endl;
  std::cout << "  -f <ext>      File extension filter (default: jpg)" << std::endl;
  std::cout << "Example: " << prog_name << " -t 4 superpoint_tf.xmodel ./imgs ./results" << std::endl;
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
  
  // Create output directory if it doesn't exist
  try {
    if (!fs::exists(output_folder)) {
      fs::create_directories(output_folder);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error creating output directory: " << e.what() << std::endl;
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
  
  std::cout << "Configuration:" << std::endl;
  std::cout << "- Number of pre/post-processing threads: " << num_threads << std::endl;
  std::cout << "- Number of DPU runners: 4 (fixed)" << std::endl;
  std::cout << "- Model: " << model_name << std::endl;
  std::cout << "- Input folder: " << input_folder << std::endl;
  std::cout << "- Output folder: " << output_folder << std::endl;
  std::cout << "- File extension: " << file_ext << std::endl;
  std::cout << "- Total images found: " << image_paths.size() << std::endl;

  try {
    // Initialize SuperPointFast
    auto superpoint = SuperPointFast(model_name, num_threads);
    
    // Create thread-safe queues
    ThreadSafeQueue<InputQueueItem> input_queue(20);
    ThreadSafeQueue<SuperPointResult> output_queue(50);

    
    // Start the SuperPointFast processor 
    superpoint.run(input_queue, output_queue);
    
    // Start producer thread to feed images when rate limiting allows
    std::thread producer_thread([&image_paths, &input_queue]() {
      size_t count = 0;
      auto start_time = std::chrono::high_resolution_clock::now();
      
      for (const auto& img_path : image_paths) {
        
        // Read image
        cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
          std::cerr << "Warning: Could not read image: " << img_path.string() << std::endl;
          continue;
        }
        
        // Create input queue item
        InputQueueItem item;
        item.index = count++;
        item.image = img;
        
        // Add to queue
        input_queue.enqueue(item);
        //there should be some delay to preserve order
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // std::cout << "Enqueued image " << count << ": " << img_path.filename().string()
        //           << " (index: " << item.index << ")" << std::endl;
      }
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      std::cout << "All " << count << " images queued in " << duration.count() << " ms" << std::endl;
      
      // Signal that no more images will be added
      input_queue.shutdown();
    });
    
    // Start consumer thread to process results
    std::thread consumer_thread([&output_queue, &output_folder]() {
      size_t result_count = 0;
      std::vector<size_t> result_indices;

      std::vector<SuperPointResult> results;
      auto start_time = std::chrono::high_resolution_clock::now();
      
      SuperPointResult result;
      while (output_queue.dequeue(result)) {
        result_count++;
        result_indices.push_back(result.index);
        results.push_back(result);
      }
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

      for(auto& res : results) {

        // Use the original image stored in the result
        cv::Mat result_img = res.img.clone();
        
        // Draw keypoints directly on the image
        for (const auto& kp : res.keypoints_cv) {
          // Draw keypoint
          cv::circle(result_img, 
                    cv::Point(kp.pt.x, kp.pt.y), 
                    3, cv::Scalar(0, 0, 255), -1);  // Red dots for keypoints
          
          // Optional: Draw small circle around keypoint for better visibility
          cv::circle(result_img, 
                    cv::Point(kp.pt.x, kp.pt.y), 
                    5, cv::Scalar(0, 255, 0), 1);  // Green circle around keypoint
        }
        
        // Save image
        std::string output_filename = output_folder + "/result_" + 
                                      std::to_string(res.index) + ".jpg";
        cv::imwrite(output_filename, result_img);

        std::cout << " (index: " << res.index 
        << ", keypoints: " << res.keypoints_cv.size() << ")" << std::endl;
      }
      
      // Check if result order matches input order
      std::cout << "Checking result order integrity..." << std::endl;
      bool is_ordered = true;
      for (size_t i = 0; i < result_indices.size(); ++i) {
        if (i != result_indices[i]) {
          std::cout << "Order mismatch at position " << i << ": found index " << result_indices[i] << std::endl;
          is_ordered = false;
        }
      }
      
      std::cout << "All " << result_count << " results processed in " << duration.count() << " ms" << std::endl;

      if (is_ordered) {
        std::cout << "Result integrity verified: all indices are in correct order." << std::endl;
      } else {
        std::cout << "WARNING: Results were processed out of order!" << std::endl;
      }
      
      // Print all indices for verification
      std::cout << "Result indices: ";
      for (size_t i = 0; i < result_indices.size(); ++i) {
        std::cout << result_indices[i] << " ";
      }
      std::cout << std::endl;
    });
    
    // Wait for all threads to complete
    producer_thread.join();
    consumer_thread.join();
    
    // Calculate statistics
    std::cout << "Processing completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}